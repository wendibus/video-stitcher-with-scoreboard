//! `MonoStitchCore` - push-first entry point for the single-input KB4
//! fisheye pipeline.
//!
//! Mirrors [`StitchCore`](super::StitchCore) (see `core/mod.rs`'s
//! module docs for the design rationale that also applies here) but
//! wraps [`MonoPipeline`] instead of [`StitchPipeline`](crate::render::pipeline::StitchPipeline)
//! and drives one raw, uncorrected fisheye camera instead of two
//! stitched ones. Deliberately does not carry replay-buffer / stacked-
//! recorder support: v1 scope is file-based mono ball tracking (load
//! one already-recorded video, track the ball, render the followed
//! crop), not live production recording. Trackers, panner, and
//! detector are the exact same trait objects
//! [`StitchCore`](super::StitchCore) uses, so `reco_autocam::setup_autocam`
//! configures either through the shared
//! [`DetectionTarget`](crate::detect::DetectionTarget) trait without
//! caring which one it got.

use std::time::Instant;

use crate::calibration::CameraParams;
use crate::detect::DetectionTarget;
use crate::detect::detector::UnifiedDetector;
use crate::detect::detector::{CameraId, ChromaFormat, Detection, DetectorFrame, RawFrame};
use crate::detect::director::{MappedDetection, ViewportPosition};
use crate::detect::panner::Panner;
use crate::detect::tracker::Tracker;
use crate::gpu::GpuContext;
use crate::gpu::rgba_readback::RgbaReadback;
use crate::projection::{kb4_mono_panorama_bounds, kb4_mono_to_panorama};
use crate::render::mono_pipeline::MonoPipeline;
use crate::render::pipeline::YuvPlanes;
use crate::render::renderer::InputFormat;
use crate::render::viewport::ViewportConfig;

use super::types::{RenderOutcome, StitchCoreError};

/// Default max angle (radians) from the optical axis a mono
/// calibration is trusted for, when the caller doesn't override it -
/// matches the widest lens [`reco_calibrate::mono_optimizer`]'s
/// multi-start grid searches (110 degrees half-FOV). Conservative by
/// construction: real lenses are usually narrower, so this rarely
/// clips a genuinely valid sample, but the flat default should be
/// tightened once a specific camera's real usable field of view is
/// known - see `kb4_mono.wgsl`'s doc comment.
pub const DEFAULT_MAX_THETA_RAD: f32 = 1.919_862_2; // 110 degrees

/// Configuration for [`MonoStitchCore::new`]. All fields have sensible
/// defaults except `camera`, `input_width`, and `input_height`.
#[derive(Debug, Clone)]
pub struct MonoStitchCoreConfig {
    /// Calibrated KB4 fisheye intrinsics for this camera (from `reco
    /// calibrate-mono`).
    pub camera: CameraParams,
    /// Max angle (radians) from the optical axis this calibration is
    /// trusted for. Defaults to [`DEFAULT_MAX_THETA_RAD`].
    pub max_theta_rad: f32,
    /// Output viewport (dimensions, FOV, rig tilt/roll).
    pub viewport: ViewportConfig,
    /// Input frame width in pixels.
    pub input_width: u32,
    /// Input frame height in pixels.
    pub input_height: u32,
    /// GPU render target format.
    pub output_format: wgpu::TextureFormat,
    /// Input pixel format (only [`InputFormat::Yuv420p`] is exercised
    /// by the CLI mono command today - CPU file decode, no zero-copy
    /// mono path yet).
    pub input_format: InputFormat,
}

impl MonoStitchCoreConfig {
    /// New config with required fields only; defaults everywhere else.
    ///
    /// `camera` is rescaled to `input_width`/`input_height` if it was
    /// calibrated at a different resolution than the video actually
    /// being processed (same camera, re-encoded/rescaled source file):
    /// `fx`/`fy`/`cx`/`cy` scale linearly with resolution, while the
    /// KB4 distortion coefficients are scale-invariant (angles, not
    /// pixels) and carry over unchanged, mirroring
    /// `reco_calibrate::lens_database`'s existing profile-rescaling
    /// logic.
    pub fn new(camera: CameraParams, input_width: u32, input_height: u32) -> Self {
        let camera = if camera.width == input_width && camera.height == input_height {
            camera
        } else {
            let scale = input_width as f64 / camera.width as f64;
            log::info!(
                "MonoStitchCoreConfig: rescaling calibration {}x{} -> {input_width}x{input_height} (scale={scale:.4})",
                camera.width,
                camera.height,
            );
            CameraParams {
                width: input_width,
                height: input_height,
                fx: camera.fx * scale,
                fy: camera.fy * scale,
                cx: camera.cx * scale,
                cy: camera.cy * scale,
                d: camera.d,
            }
        };
        Self {
            camera,
            max_theta_rad: DEFAULT_MAX_THETA_RAD,
            viewport: ViewportConfig {
                width: 1920,
                height: 1080,
                fov_degrees: 75.0,
                blend_width: 0.0,
                rig_tilt: 0.0,
                rig_roll: 0.0,
                lens_correction_amount: 1.0,
            },
            input_width,
            input_height,
            output_format: wgpu::TextureFormat::Rgba8Unorm,
            input_format: InputFormat::Yuv420p,
        }
    }
}

/// Canonical push-first mono stitching core.
///
/// See the module docs for design rationale.
pub struct MonoStitchCore {
    pipeline: MonoPipeline,
    readback: RgbaReadback,
    output_width: u32,
    output_height: u32,

    ball_tracker: Option<Box<dyn Tracker>>,
    player_tracker: Option<Box<dyn Tracker>>,
    panner: Option<Box<dyn Panner>>,
    previous_panner_pose: ViewportPosition,

    detector: Option<Box<dyn UnifiedDetector>>,
    detection_interval: u64,
    last_detections: Vec<MappedDetection>,

    /// Built once at construction and reused every frame - see
    /// [`crate::calibration::MatchCalibration::mono_placeholder`]'s
    /// docs for why `DispatchContext` needs one at all here.
    placeholder_calibration: crate::calibration::MatchCalibration,

    frame_count: u64,
    session_start: Option<Instant>,

    /// Panorama-space (yaw, pitch) range the camera's calibrated field
    /// of view actually covers - the *un-inset* raw painted-region
    /// bounds, computed once (camera + `max_theta_rad` never change
    /// after construction). [`Self::clamp_pose_for_fov`] insets this
    /// per-frame for whatever FOV the panner requests that frame,
    /// since a wider FOV needs a larger inset margin (see its doc
    /// comment) - this cannot be precomputed once the FOV itself can
    /// vary frame to frame.
    raw_yaw_bounds: (f32, f32),
    raw_pitch_bounds: (f32, f32),
}

impl MonoStitchCore {
    /// Build a new mono core. Owns the supplied [`GpuContext`].
    pub fn new(gpu: GpuContext, config: MonoStitchCoreConfig) -> Result<Self, StitchCoreError> {
        let output_width = config.viewport.width;
        let output_height = config.viewport.height;

        let (raw_yaw_bounds, raw_pitch_bounds) =
            kb4_mono_panorama_bounds(&config.camera, config.max_theta_rad);

        let pipeline = MonoPipeline::with_gpu(
            gpu,
            config.camera,
            config.max_theta_rad,
            config.viewport,
            config.input_width,
            config.input_height,
            config.output_format,
            config.input_format,
        )?;

        let readback = RgbaReadback::new(pipeline.gpu(), output_width, output_height)?;

        Ok(Self {
            pipeline,
            readback,
            output_width,
            output_height,
            ball_tracker: None,
            player_tracker: None,
            panner: None,
            previous_panner_pose: ViewportPosition::default(),
            detector: None,
            detection_interval: 1,
            last_detections: Vec::new(),
            placeholder_calibration: crate::calibration::MatchCalibration::mono_placeholder(),
            frame_count: 0,
            session_start: None,
            raw_yaw_bounds,
            raw_pitch_bounds,
        })
    }

    /// Inset [`Self::raw_yaw_bounds`]/[`Self::raw_pitch_bounds`] by the
    /// output viewport's *corner* half-angle for the given `fov_degrees`,
    /// so a pose clamped to the result keeps the *entire rendered
    /// frame* inside the camera's calibrated field of view - not just
    /// its center ray - eliminating the black out-of-coverage wedges a
    /// pose too close to the edge of the camera's field of view
    /// produces. Takes `fov_degrees` as a parameter (rather than being
    /// precomputed once) because the panner can request a different
    /// FOV every frame (see [`Self::clamp_pose_for_fov`]) - a wider
    /// requested FOV needs a proportionally larger inset margin, so
    /// the safe yaw/pitch range genuinely shrinks as the panner zooms
    /// out, and grows as it zooms in.
    ///
    /// Uses the corner angle (`atan(hypot(tan_half_h, tan_half_v))`),
    /// not the smaller edge-midpoint angles (`half_h_fov`/`half_v_fov`
    /// alone) - a rectangular viewport's corners sit further from its
    /// center than either edge's midpoint, since a corner combines
    /// both the horizontal *and* vertical offset simultaneously. Only
    /// margining by the edge angles (an earlier version of this
    /// function did) under-protects exactly the corners, which is
    /// what real-footage testing showed as a visible black-corner
    /// artifact even with pose nowhere near `yaw_min`/`yaw_max` alone.
    ///
    /// Collapses to the midpoint on either axis where the requested
    /// FOV is wider than the available coverage (nothing sensible to
    /// clamp to in that degenerate case - matches the L-shape coverage
    /// code's "collapse to midpoint if bounds inverted" fallback).
    fn inset_bounds_for_fov(
        &self,
        fov_degrees: f32,
        output_width: u32,
        output_height: u32,
    ) -> ((f32, f32), (f32, f32)) {
        let (yaw_min, yaw_max) = self.raw_yaw_bounds;
        let (pitch_min, pitch_max) = self.raw_pitch_bounds;

        let half_v_fov = fov_degrees.to_radians() * 0.5;
        let aspect = output_width as f32 / output_height.max(1) as f32;
        let tan_half_v = half_v_fov.tan();
        let tan_half_h = tan_half_v * aspect;
        let corner_margin = tan_half_h.hypot(tan_half_v).atan();

        // Defense in depth against NaN reaching `resolve_current_pose`'s
        // `clamp` call (which panics on NaN bounds): `kb4_mono_panorama_bounds`
        // already guards its own degenerate case, but `lo + hi` below can
        // still be NaN if `lo`/`hi` were ever +-infinity from some other
        // caller/future change - fall back to a fixed dead-ahead point
        // rather than trust an unchecked midpoint.
        let inset = |lo: f32, hi: f32, margin: f32| -> (f32, f32) {
            let (lo, hi) = (lo + margin, hi - margin);
            if lo <= hi {
                (lo, hi)
            } else {
                let mid = (lo + hi) * 0.5;
                if mid.is_finite() { (mid, mid) } else { (0.0, 0.0) }
            }
        };
        (
            inset(yaw_min, yaw_max, corner_margin),
            inset(pitch_min, pitch_max, corner_margin),
        )
    }

    /// Clamp a prospective pose (yaw/pitch/optional FOV request) so
    /// the *entire rendered frame* stays inside this camera's
    /// calibrated coverage - the mono counterpart of
    /// [`StitchCore::safe_clamp`](super::StitchCore::safe_clamp) /
    /// `pose.rs`'s `resolve_current_pose`. `pose.fov_degrees: None`
    /// (no dynamic-zoom panner attached, or it chose not to request a
    /// change this frame) falls back to the pipeline's current FOV.
    fn clamp_pose_for_fov(&self, pose: ViewportPosition) -> ViewportPosition {
        let fov = pose.fov_degrees.unwrap_or_else(|| self.pipeline.fov());
        let (yaw_bounds, pitch_bounds) =
            self.inset_bounds_for_fov(fov, self.output_width, self.output_height);
        ViewportPosition {
            yaw: pose.yaw.clamp(yaw_bounds.0, yaw_bounds.1),
            pitch: pose.pitch.clamp(pitch_bounds.0, pitch_bounds.1),
            fov_degrees: Some(fov),
        }
    }

    // -----------------------------------------------------------------
    // Tracker / panner wiring (identical contract to StitchCore)
    // -----------------------------------------------------------------

    /// Attach a singleton ball tracker. Replaces any existing one.
    pub fn set_ball_tracker(&mut self, tracker: Box<dyn Tracker>) {
        log::info!(
            "MonoStitchCore: ball tracker attached (class_id={})",
            tracker.class_id()
        );
        self.ball_tracker = Some(tracker);
    }

    /// Attach a multi-entity player tracker. Replaces any existing one.
    pub fn set_player_tracker(&mut self, tracker: Box<dyn Tracker>) {
        log::info!(
            "MonoStitchCore: player tracker attached (class_id={})",
            tracker.class_id()
        );
        self.player_tracker = Some(tracker);
    }

    /// Attach a panner. Replaces any existing one.
    pub fn set_panner(&mut self, panner: Box<dyn Panner>) {
        log::info!("MonoStitchCore: panner attached");
        self.panner = Some(panner);
    }

    /// Attach a unified-trait detector. Replaces any existing one.
    pub fn set_detector(&mut self, detector: Box<dyn UnifiedDetector>) {
        self.detector = Some(detector);
    }

    /// Set how often detection runs (`1` = every frame).
    pub fn set_detection_interval(&mut self, interval: u64) {
        self.detection_interval = interval.max(1);
    }

    /// The GPU context owning every resource.
    pub fn gpu(&self) -> &GpuContext {
        self.pipeline.gpu()
    }

    /// Shared access to the underlying pipeline.
    pub fn pipeline(&self) -> &MonoPipeline {
        &self.pipeline
    }

    /// Number of frames submitted so far.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Output dimensions in pixels.
    pub fn output_dims(&self) -> (u32, u32) {
        (self.output_width, self.output_height)
    }

    // -----------------------------------------------------------------
    // Submit / render
    // -----------------------------------------------------------------

    /// Flush one pending frame from the triple-buffered readback.
    ///
    /// Call in a loop after the last [`Self::submit_frame_yuv`] to
    /// drain the frames still in flight (the readback ring lags two
    /// frames behind submission - see that method's docs). Returns
    /// `None` once nothing is pending.
    pub fn flush_pending(&mut self) -> Result<Option<&[u8]>, StitchCoreError> {
        Ok(self.readback.flush_pending(self.pipeline.gpu())?)
    }

    /// Submit a mono YUV420P frame and render the current pose.
    ///
    /// Same triple-buffered readback contract as
    /// [`StitchCore::submit_frame_yuv`](super::StitchCore::submit_frame_yuv):
    /// the first two calls produce [`RenderOutcome::Warmup`]; from the
    /// third call onward every submit yields RGBA bytes from two
    /// frames ago.
    pub fn submit_frame_yuv(
        &mut self,
        frame: &YuvPlanes<'_>,
    ) -> Result<RenderOutcome<'_>, StitchCoreError> {
        if self.session_start.is_none() {
            self.session_start = Some(Instant::now());
        }

        let ran_detection = self.detector.is_some() && self.should_run_detection();
        if ran_detection {
            let (src_w, src_h) = self.pipeline.source_info();
            let dets = self.run_yuv_detection(frame, src_w, src_h);
            self.last_detections = self.map_detections_to_panorama(dets);
        }

        let pose = self.resolve_current_pose();
        let cmd = self
            .pipeline
            .render_to_target(frame, pose.yaw, pose.pitch)?;
        let rgba =
            self.readback
                .readback(self.pipeline.gpu(), self.pipeline.render_target(), cmd)?;
        self.frame_count += 1;
        Ok(match rgba {
            Some(bytes) => RenderOutcome::Rgba(bytes),
            None => RenderOutcome::Warmup,
        })
    }

    fn should_run_detection(&self) -> bool {
        self.detection_interval > 0 && self.frame_count.is_multiple_of(self.detection_interval)
    }

    /// Run the attached detector against a mono YUV420P frame.
    ///
    /// Tags every detection [`CameraId::Left`] - the sentinel this
    /// single-camera path uses everywhere a `CameraId` is structurally
    /// required (trackers/panners are class-based, not camera-count-
    /// aware, so this has no behavioral effect - see the module docs).
    fn run_yuv_detection(
        &mut self,
        frame: &YuvPlanes<'_>,
        source_width: u32,
        source_height: u32,
    ) -> Vec<Detection> {
        let Some(ref mut detector) = self.detector else {
            return Vec::new();
        };
        let raw = RawFrame {
            y: frame.y,
            chroma: ChromaFormat::Yuv420p {
                u: frame.u,
                v: frame.v,
            },
            width: source_width,
            height: source_height,
        };
        match detector.detect(CameraId::Left, &DetectorFrame::Cpu(raw)) {
            Ok(v) => v,
            Err(e) => {
                log::warn!("MonoStitchCore detector '{}': {e}", detector.name());
                Vec::new()
            }
        }
    }

    /// Map raw camera-space detections to panorama-space
    /// [`MappedDetection`]s the director can consume, via
    /// [`kb4_mono_to_panorama`] (the mono counterpart of
    /// [`crate::projection::camera_to_panorama`]). A detection whose
    /// inverse-KB4 solve doesn't converge (rare - see that function's
    /// docs) gets `position: None`, which downstream trackers already
    /// treat as "drop this detection" (e.g. `BallTracker`'s filter
    /// chain).
    fn map_detections_to_panorama(&self, detections: Vec<Detection>) -> Vec<MappedDetection> {
        let camera = self.pipeline.camera();
        detections
            .into_iter()
            .map(|d| {
                let position = kb4_mono_to_panorama(d.center_x, d.center_y, camera);
                MappedDetection {
                    camera: d.camera,
                    class_id: d.class_id,
                    confidence: d.confidence,
                    camera_center: (d.center_x, d.center_y),
                    camera_size: (d.width, d.height),
                    position,
                }
            })
            .collect()
    }

    /// Resolve this frame's render pose from the attached panner, then
    /// clamp it via [`Self::clamp_pose_for_fov`] - the panner (dead-
    /// zone/lookahead/velocity/dynamic-zoom logic in
    /// [`reco_autocam::panners::FieldPanner`]) has no notion of "the
    /// projection runs out of painted video past here," so without
    /// this clamp a ball tracked near the edge of the source's angular
    /// sweep (or a panner zooming in tight enough that the frame edge
    /// reaches past coverage) drives the rendered crop straight into
    /// the shader's out-of-coverage transparent (black-once-encoded)
    /// region. Also writes the resolved FOV back onto the pipeline
    /// (mirrors `StitchCore`'s `pose.rs`) so the upcoming render
    /// actually uses whatever zoom level the panner requested this
    /// frame, instead of a fixed FOV frozen at construction time.
    fn resolve_current_pose(&mut self) -> ViewportPosition {
        let timestamp_ms = self
            .session_start
            .map(|s| s.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);

        let raw = crate::detect::panner::dispatch(
            self.panner.as_mut(),
            self.player_tracker.as_mut(),
            self.ball_tracker.as_mut(),
            &mut self.previous_panner_pose,
            None,
            &[],
            crate::detect::panner::DispatchContext {
                detections: &self.last_detections,
                calibration: &self.placeholder_calibration,
                frame_index: self.frame_count,
                timestamp_ms,
                caller: "MonoStitchCore",
            },
        )
        .map(|r| r.pose)
        .unwrap_or_default();

        let clamped = self.clamp_pose_for_fov(raw);
        if let Some(fov) = clamped.fov_degrees {
            self.pipeline.set_fov(fov);
        }
        clamped
    }
}

impl DetectionTarget for MonoStitchCore {
    fn set_detector(&mut self, detector: Box<dyn UnifiedDetector>) {
        self.set_detector(detector);
    }
    fn set_detection_interval(&mut self, interval: u64) {
        self.set_detection_interval(interval);
    }
    fn set_ball_tracker(&mut self, tracker: Box<dyn Tracker>) {
        self.set_ball_tracker(tracker);
    }
    fn set_player_tracker(&mut self, tracker: Box<dyn Tracker>) {
        self.set_player_tracker(tracker);
    }
    fn set_panner(&mut self, panner: Box<dyn Panner>) {
        self.set_panner(panner);
    }
    fn source_info(&self) -> (u32, u32) {
        self.pipeline.source_info()
    }
    fn gpu(&self) -> &GpuContext {
        self.gpu()
    }
}
