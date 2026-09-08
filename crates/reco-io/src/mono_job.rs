//! One-shot file-to-file mono (single-camera, KB4 fisheye projection)
//! stitching (Layer 3 API) - mirrors [`crate::stitch_job::StitchJob`]
//! for [`reco_core::core::mono::MonoStitchCore`] instead of
//! [`reco_core::session::StitchSession`].
//!
//! Deliberately smaller than `StitchJob`: no lookahead-buffered
//! smoothing, replay recording, chained segments, or pipeline event
//! export - v1 scope is "load one already-recorded single-camera
//! video, track the ball, render the followed crop", not live
//! production recording. The CPU decode path only (no GPU zero-copy):
//! that path exists for live-camera capture, which mono doesn't serve
//! yet.
//!
//! Unlike [`StitchJob`](crate::stitch_job::StitchJob), this takes the
//! camera calibration as an in-memory [`CameraParams`] rather than a
//! JSON file path: the mono calibration file format
//! (`reco calibrate-mono`'s output) is a `reco-calibrate` concern, and
//! `reco-io` can't depend on `reco-calibrate` (which already depends
//! on `reco-io`) - so parsing that file lives in `reco-cli`, which
//! depends on both.
//!
//! # Example
//!
//! ```rust,ignore
//! use reco_io::MonoJob;
//! use reco_io::output::{Codec, Quality};
//! use reco_core::calibration::CameraParams;
//!
//! let camera: CameraParams = /* loaded from `reco calibrate-mono`'s output */;
//! MonoJob::new("wide.mp4", "output.mp4", camera)
//!     .codec(Codec::HEVC)
//!     .quality(Quality::High)
//!     .on_session(|core, fps| {
//!         // attach detector/tracker/panner via reco_autocam::setup_autocam
//!     })
//!     .run(&interrupted)?;
//! ```

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use reco_core::calibration::CameraParams;
use reco_core::core::mono::{MonoStitchCore, MonoStitchCoreConfig};
use reco_core::core::types::RenderOutcome;
use reco_core::render::planes::YuvPlanes;
use reco_core::render::renderer::InputFormat;

use crate::ffmpeg::decoder::VideoDecoder;
use crate::ffmpeg::encoder::{EncoderConfig, VideoEncoder};
use crate::output::{Bitrate, Codec, Format, Quality};

/// Errors from [`MonoJob::run`].
#[derive(Debug, thiserror::Error)]
pub enum MonoJobError {
    /// Source video could not be opened or decoded.
    #[error("decode: {0}")]
    Decode(#[from] crate::ffmpeg::decoder::DecodeError),
    /// GPU initialization failed.
    #[error("GPU: {0}")]
    Gpu(#[from] reco_core::gpu::GpuError),
    /// Mono pipeline construction or render error.
    #[error("stitch core: {0}")]
    Core(#[from] reco_core::core::types::StitchCoreError),
    /// Encoder error.
    #[error("encoder: {0}")]
    Encoder(#[from] crate::ffmpeg::encoder::EncodeError),
}

/// Result summary from a completed [`MonoJob::run`].
#[derive(Debug, Clone, Copy)]
pub struct MonoJobResult {
    /// Number of frames encoded to the output.
    pub frame_count: u64,
}

/// One-shot mono stitching job: one video file in, encoded video out.
pub struct MonoJob {
    input: PathBuf,
    output: PathBuf,
    camera: CameraParams,
    max_theta_rad: f32,
    rig_tilt: f32,
    rig_roll: f32,

    codec: Codec,
    bitrate: Bitrate,
    format: Format,
    resolution: Option<(u32, u32)>,
    encoder_name: Option<String>,
    quality_value: Option<u8>,
    preset: Option<String>,
    with_audio: bool,

    detection_interval: u64,

    #[allow(clippy::type_complexity)]
    session_hook: Option<Box<dyn FnOnce(&mut MonoStitchCore, f64) + Send>>,
}

impl MonoJob {
    /// New job with required fields only; defaults everywhere else.
    ///
    /// `camera` is this camera's calibrated KB4 fisheye intrinsics
    /// (from `reco calibrate-mono`'s output) - rescaled automatically
    /// to the source video's actual resolution if it differs from the
    /// resolution the calibration frame was extracted at, see
    /// [`MonoStitchCoreConfig::new`].
    pub fn new(input: impl Into<PathBuf>, output: impl Into<PathBuf>, camera: CameraParams) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
            camera,
            max_theta_rad: reco_core::core::mono::DEFAULT_MAX_THETA_RAD,
            rig_tilt: 0.0,
            rig_roll: 0.0,
            codec: Codec::default(),
            bitrate: Bitrate::default(),
            format: Format::default(),
            resolution: None,
            encoder_name: None,
            quality_value: None,
            preset: None,
            with_audio: true,
            detection_interval: 1,
            session_hook: None,
        }
    }

    /// Override the max angle (radians) from the optical axis this
    /// calibration is trusted for. Defaults to
    /// [`reco_core::core::mono::DEFAULT_MAX_THETA_RAD`].
    pub fn max_theta_rad(mut self, radians: f32) -> Self {
        self.max_theta_rad = radians;
        self
    }

    /// Rig tilt/roll correction (radians) for a physically un-level
    /// mount - same convention as the stereo path's `view_matrix`.
    /// Defaults to `(0.0, 0.0)`; a future calibration-derived default
    /// (from the solved camera pose) is a planned follow-up, not yet
    /// wired in - see `MonoStitchCoreConfig`'s module docs.
    pub fn rig_tilt_roll(mut self, tilt: f32, roll: f32) -> Self {
        self.rig_tilt = tilt;
        self.rig_roll = roll;
        self
    }

    /// Output video codec.
    pub fn codec(mut self, codec: Codec) -> Self {
        self.codec = codec;
        self
    }

    /// Encoder-agnostic quality tier.
    pub fn quality(mut self, quality: Quality) -> Self {
        self.bitrate = Bitrate::Quality(quality);
        self
    }

    /// Output resolution. Defaults to 1920x1080.
    pub fn resolution(mut self, width: u32, height: u32) -> Self {
        self.resolution = Some((width, height));
        self
    }

    /// Force a specific encoder by name.
    pub fn encoder_name(mut self, name: impl Into<String>) -> Self {
        self.encoder_name = Some(name.into());
        self
    }

    /// Normalized quality override (0-100), replacing the quality preset.
    pub fn quality_value(mut self, quality: u8) -> Self {
        self.quality_value = Some(quality);
        self
    }

    /// Override the encoder preset string.
    pub fn preset(mut self, preset: impl Into<String>) -> Self {
        self.preset = Some(preset.into());
        self
    }

    /// Copy audio from the input file into the output (default: on).
    pub fn with_audio(mut self, enabled: bool) -> Self {
        self.with_audio = enabled;
        self
    }

    /// Run detection every N frames (default: 1 = every frame).
    pub fn detection_interval(mut self, interval: u64) -> Self {
        self.detection_interval = interval.max(1);
        self
    }

    /// Register a callback that configures the [`MonoStitchCore`]
    /// (detector/tracker/panner) before the frame loop starts - the
    /// mono counterpart of [`crate::stitch_job::StitchJob::on_session`].
    /// Receives the source's fps (only known once the input is open),
    /// since [`reco_autocam::setup_autocam`] needs it.
    pub fn on_session(
        mut self,
        hook: impl FnOnce(&mut MonoStitchCore, f64) + Send + 'static,
    ) -> Self {
        self.session_hook = Some(Box::new(hook));
        self
    }

    /// Run the job: decode, track, render, encode, until the source is
    /// exhausted or `interrupted` is set.
    pub fn run(mut self, interrupted: &AtomicBool) -> Result<MonoJobResult, MonoJobError> {
        crate::init();

        let mut decoder = VideoDecoder::open(&self.input)?;
        let (input_width, input_height) = (decoder.width(), decoder.height());
        let (out_w, out_h) = self.resolution.unwrap_or((1920, 1080));

        let gpu = reco_core::gpu::GpuContext::new_blocking()?;
        let mut config = MonoStitchCoreConfig::new(self.camera, input_width, input_height);
        config.max_theta_rad = self.max_theta_rad;
        config.viewport.width = out_w;
        config.viewport.height = out_h;
        config.viewport.rig_tilt = self.rig_tilt;
        config.viewport.rig_roll = self.rig_roll;
        config.input_format = InputFormat::Yuv420p;
        let mut core = MonoStitchCore::new(gpu, config)?;

        let fps_rational = {
            let r = decoder.frame_rate();
            if r.0 > 0 && r.1 > 0 {
                (r.0, r.1)
            } else {
                (30, 1)
            }
        };
        let fps = fps_rational.0 as f64 / fps_rational.1 as f64;

        if let Some(hook) = self.session_hook.take() {
            hook(&mut core, fps);
        }

        let quality = match &self.bitrate {
            Bitrate::Quality(q) => *q,
            Bitrate::Crf(_) => Quality::Balanced,
        };
        let enc_config = EncoderConfig {
            encoder_name: self.encoder_name.clone(),
            codec: self.codec.into(),
            quality_preset: quality.into(),
            quality: self.quality_value,
            preset: self.preset.clone(),
            audio_source: self.with_audio.then(|| vec![self.input.clone()]),
            audio_start_time: 0.0,
            container: self.format.into(),
            gop_size: None,
            stream_url: None,
        };
        let mut encoder =
            VideoEncoder::new(&self.output, out_w, out_h, fps_rational.into(), &enc_config)?;

        core.set_detection_interval(self.detection_interval);

        let mut frame_count = 0u64;
        while !interrupted.load(Ordering::Relaxed) {
            let Some(frame) = decoder.next_frame()? else {
                break;
            };
            let planes = YuvPlanes {
                y: &frame.y,
                u: &frame.u,
                v: &frame.v,
            };
            match core.submit_frame_yuv(&planes)? {
                RenderOutcome::Rgba(rgba) => {
                    encoder.write_frame(rgba)?;
                    frame_count += 1;
                }
                RenderOutcome::Warmup => {}
            }
        }

        // Drain the triple-buffered readback so the last couple of
        // submitted frames (still in flight when the source ran out)
        // make it into the output instead of being silently dropped.
        while let Some(rgba) = core.flush_pending()? {
            encoder.write_frame(rgba)?;
            frame_count += 1;
        }

        encoder.finish()?;

        log::info!(
            "MonoJob: {frame_count} frames -> {} ({:.1}fps source)",
            self.output.display(),
            fps
        );
        Ok(MonoJobResult { frame_count })
    }
}
