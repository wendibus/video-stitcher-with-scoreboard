//! Coordinate mapping between camera pixel space and panoramic viewport.
//!
//! These functions bridge detection coordinates (in individual camera frames)
//! and virtual camera orientation (yaw/pitch), enabling:
//! - **Detection mapping**: convert detector output to director yaw/pitch
//! - **"No-black" panning**: compute valid viewport bounds to avoid black edges
//!
//! ## Coordinate Spaces
//!
//! ```text
//! Camera pixel [0,1]  ──undistort──►  Plane UV  ──model matrix──►  World 3D
//!                                                                       │
//! Virtual camera yaw/pitch  ◄──decompose──  Direction from camera  ◄────┘
//! ```

mod coverage;
mod geometry;
mod virtual_camera;

// Re-export coverage types so external code can still use
// `crate::projection::CoverageBoundary` etc.
pub use coverage::{ClampedPosition, CoverageBoundary, PanoramaExtent};

// Re-export geometry utility.
pub use geometry::point_in_polygon;

// Re-export virtual camera (pub(crate) visibility preserved).
pub(crate) use virtual_camera::VirtualCamera;

use crate::calibration::{CameraParams, MatchCalibration};
use crate::detect::detector::CameraId;
use crate::detect::director::ViewportPosition;
use crate::render::scene::SceneGeometry;

use nalgebra::{Point3, Vector3};

// ---------------------------------------------------------------------------
// M3 foundation: Projection trait + LShapeProjection marker.
// ---------------------------------------------------------------------------
//
// Plan-execution §2.5 + §7 decision 8: the future StitchCore takes a
// `Box<dyn Projection>` instead of hardcoding the 2-plane L-shape
// geometry. This makes alt-projections (cylindrical / flat-mixing-
// shader / mono-single-plane / equirect / N-camera panoramic) drop-in
// additions later without reshaping the core session API.
//
// This commit lands the trait + a marker implementation for today's
// L-shape geometry. It does NOT move the existing `camera_to_panorama`
// etc. free functions into the trait - that migration happens when
// StitchCore is being written and the real method set emerges from
// usage. Landing the shape first lets parallel design work on a
// second projection (§7 decision 8, user-chosen form) start without
// re-plumbing reco-core.

/// A panoramic projection geometry.
///
/// Implemented by concrete projections (today's 2-plane L-shape,
/// future cylindrical / flat-mix / mono / N-camera). Dispatched
/// dynamically by StitchCore so swapping projections at session
/// construction time does not require recompilation.
///
/// # Bounds
///
/// `Send + Sync` because StitchCore stores projections behind a
/// shared reference and the render thread reads them concurrently.
pub trait Projection: Send + Sync {
    /// Short human-readable name for logs + diagnostic bundles.
    fn name(&self) -> &'static str;

    /// Number of input cameras this projection consumes. 1 for mono,
    /// 2 for today's L-shape stereo, N>2 for future panoramic rigs.
    fn camera_count(&self) -> u8;

    /// WGSL fragment shader source for the composite pass that
    /// transforms per-camera undistorted textures into the final
    /// panorama output.
    ///
    /// Returned as a string so wgpu can compile it at pipeline
    /// creation. Today's L-shape geometry returns an empty string:
    /// its shader is still embedded in `stitch_renderer.rs`. The
    /// migration happens when StitchCore takes over rendering and
    /// dispatches composite via this trait.
    fn wgsl_composite_source(&self) -> &str {
        ""
    }
}

/// Marker type for today's 2-plane L-shape stereo projection.
///
/// The geometry is documented in [`scene::SceneGeometry`](crate::render::scene::SceneGeometry).
/// All the real math still lives in the free functions below and in
/// `stitch_renderer.rs`; this struct carries no state today. It is
/// here to make StitchCore's `Box<dyn Projection>` slot have a
/// concrete default that matches shipping behavior.
#[derive(Debug, Default, Clone, Copy)]
pub struct LShapeProjection;

impl Projection for LShapeProjection {
    fn name(&self) -> &'static str {
        "l-shape-stereo-2camera"
    }

    fn camera_count(&self) -> u8 {
        2
    }
}

// ---------------------------------------------------------------------------
// Cylindrical single-input projection (plan step 9, second Projection impl).
// ---------------------------------------------------------------------------
//
// Models a single video as a texture painted on the inside of a
// cylinder of radius `focal_length`. The virtual camera sits on the
// cylinder axis and looks outward; pan/tilt/zoom rotate the camera and
// scale FOV. Matches the `gilbertchen/actionstitch-player` projection
// (MIT-licensed 180-degree cylindrical video player) enough that
// calibration files from that ecosystem could be consumed with small
// adapter code.
//
// Design goal of landing this here (not just as a shader file):
// proves the plan's claim that the `Projection` trait supports
// camera_count() != 2 so future mono / N-camera / alt-projection
// impls can plug in without reshaping StitchCore's API. Ships with:
//
//   - A configurable `CylindricalProjection` with defaults that
//     mirror actionstitch's (focal_length=2400, sweep=PI = 180deg,
//     screen_rotation=0, video_height sourced from the input).
//   - A WGSL shader at `shaders/cylindrical_mono.wgsl` returned
//     verbatim from `wgsl_composite_source()`.
//   - camera_count() = 1.
//
// Deliberately NOT wired into `StitchCore` / `StitchPipeline` in this
// commit - that migration is a follow-up that needs a mono submit
// path (`submit_frame_yuv_mono`) and a different bind group layout.
// The trait-side contract is the deliverable here.

/// Configuration for a [`CylindricalProjection`]. Defaults match the
/// `actionstitch-player` projection (180-degree sweep, 2400px focal
/// length, no screen rotation).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CylindricalProjectionConfig {
    /// Cylinder radius in world units. Larger values = narrower
    /// cylindrical wrap per pixel, so the panorama feels flatter.
    /// `actionstitch` defaults to `2400` and exposes a slider from
    /// 1000 to 5000.
    pub focal_length: f32,
    /// Full horizontal angular sweep in radians. `std::f32::consts::PI`
    /// (180 degrees) is the canonical action-camera case; 2π would be
    /// a full 360-degree cylinder.
    pub angular_sweep_rad: f32,
    /// Screen tilt around the view axis in radians. `actionstitch`
    /// exposes this as a ±30-degree slider labelled "Screen tilt" and
    /// uses it to correct for a rig that is not level side-to-side.
    pub screen_rotation_rad: f32,
    /// Video height in world units, on the *same* scale as
    /// `focal_length` - NOT independently normalized. The bare
    /// default (`1.0`) only makes sense paired with a tiny
    /// `focal_length`; for the real default `focal_length = 2400.0`
    /// it is almost always wrong (the painted patch collapses to a
    /// sliver, and pitch resolution goes to ~0). Callers that know the
    /// source video's pixel dimensions should compute this via
    /// [`CylindricalProjectionConfig::video_height_for_aspect`]
    /// instead of leaving the default in place.
    pub video_height: f32,
}

impl Default for CylindricalProjectionConfig {
    fn default() -> Self {
        Self {
            focal_length: 2400.0,
            angular_sweep_rad: std::f32::consts::PI,
            screen_rotation_rad: 0.0,
            video_height: 1.0,
        }
    }
}

/// Single-input cylindrical projection.
///
/// Consumes one camera (`camera_count() == 1`) and renders it as if
/// painted on the inside of a cylinder of radius `config.focal_length`.
/// The virtual camera sits on the cylinder axis.
///
/// Attribution: the projection geometry (focal-length, angular-sweep,
/// and screen-rotation tilt) is the one used by
/// `gilbertchen/actionstitch-player` (MIT-licensed 180-degree
/// cylindrical video player). The WGSL shader here is a from-scratch
/// reimplementation of that model for wgpu; no code is copied.
#[derive(Debug, Clone, Copy, Default)]
pub struct CylindricalProjection {
    /// Projection parameters. `Default` uses `actionstitch`-matching
    /// values (see [`CylindricalProjectionConfig::default`]).
    pub config: CylindricalProjectionConfig,
}

impl CylindricalProjectionConfig {
    /// The `video_height` (world units) that paints the source video
    /// onto the cylinder without vertical stretch/squash, given its
    /// pixel aspect ratio.
    ///
    /// The painted patch's horizontal extent (arc length at radius
    /// `focal_length`, swept through `angular_sweep_rad`) is
    /// `focal_length * angular_sweep_rad`. Scaling `video_height` to
    /// match that arc length times the source's height/width ratio
    /// keeps the two axes in the same world-unit scale - `focal_length`
    /// and the bare default `video_height = 1.0` are not automatically
    /// consistent (see [`Self::default`]'s doc comment), so a caller
    /// that knows the source dimensions should always compute this
    /// instead of leaving `video_height` at its placeholder default.
    /// [`crate::core::mono::MonoStitchCoreConfig::new`] does exactly
    /// that.
    pub fn video_height_for_aspect(
        focal_length: f32,
        angular_sweep_rad: f32,
        input_width: u32,
        input_height: u32,
    ) -> f32 {
        let arc_length = focal_length * angular_sweep_rad;
        arc_length * (input_height as f32 / input_width as f32)
    }
}

impl CylindricalProjection {
    /// Build a new cylindrical projection with the given config.
    pub fn new(config: CylindricalProjectionConfig) -> Self {
        Self { config }
    }

    /// Compute the cylinder's `theta_start` angle in radians:
    /// `PI/2 - angular_sweep/2`. This is where the video's left edge
    /// lands on the cylinder surface and matches actionstitch's
    /// `THREE.CylinderGeometry(..., Math.PI / 2 - s / 2, s)`.
    pub fn theta_start_rad(&self) -> f32 {
        std::f32::consts::FRAC_PI_2 - self.config.angular_sweep_rad * 0.5
    }
}

/// WGSL source for the cylindrical-mono composite pass. Embedded at
/// compile time so `wgsl_composite_source()` can return `&'static str`.
const CYLINDRICAL_MONO_WGSL: &str = include_str!("../shaders/cylindrical_mono.wgsl");

impl Projection for CylindricalProjection {
    fn name(&self) -> &'static str {
        "cylindrical-mono-1camera"
    }

    fn camera_count(&self) -> u8 {
        1
    }

    fn wgsl_composite_source(&self) -> &str {
        CYLINDRICAL_MONO_WGSL
    }
}

/// Stand-in "camera position" for [`VirtualCamera`] basis construction
/// on the cylindrical/mono path.
///
/// The cylindrical camera genuinely sits at the world origin (on the
/// cylinder axis) - but [`VirtualCamera::new`] derives its rest-forward
/// direction as `(-eye).normalize()`, which is `(0,0,0).normalize()`
/// (NaN) for an eye exactly at the origin. That convention exists for
/// the L-shape path, where cameras are positioned away from the origin
/// and look inward at it; the cylindrical camera has no such "look at
/// the origin" relationship; It only needs *some* fixed, non-degenerate
/// rest-forward direction to decompose yaw/pitch against. `[0,0,-1]`
/// picks rest-forward `= (0,0,1)`, matching this module's
/// `cylindrical_to_panorama`/[`super::mono_renderer`]'s `view_matrix`
/// call sites, which both use it - keeping it as one named constant
/// instead of a magic literal in two places prevents them drifting out
/// of sync.
pub(crate) const MONO_CAMERA_REST_POSITION: [f32; 3] = [0.0, 0.0, -1.0];

/// Map a detection in cylindrical-camera pixel space to the yaw/pitch
/// needed to center the virtual camera on it.
///
/// Exact inverse of `cylindrical_mono.wgsl`'s `fs_cylindrical_mono`
/// forward sampling (video UV -> cylinder hit -> ray): given a
/// normalized detection center, reconstruct the cylinder hit point and
/// decompose the camera-at-origin direction to it into yaw/pitch via
/// [`direction_to_yaw_pitch`] - the same basis every other panorama
/// consumer (panners, directors, [`camera_to_panorama`]) shares, just
/// with the camera pinned to the cylinder axis (`[0,0,0]`) instead of
/// an L-shape plane's calibrated position.
///
/// `norm_x`/`norm_y` are in normalized `[0.0, 1.0]` image coordinates
/// (as returned by [`Detection`](crate::detect::detector::Detection)).
pub fn cylindrical_to_panorama(
    norm_x: f32,
    norm_y: f32,
    config: &CylindricalProjectionConfig,
) -> ViewportPosition {
    let theta_norm = norm_x;
    let v_norm = 1.0 - norm_y;
    let half_sweep = config.angular_sweep_rad * 0.5;
    let theta_start = std::f32::consts::FRAC_PI_2 - half_sweep;
    let theta = theta_start + theta_norm * config.angular_sweep_rad;
    let y_world = v_norm * config.video_height - config.video_height * 0.5;

    let hit = Vector3::new(
        config.focal_length * theta.cos(),
        y_world,
        config.focal_length * theta.sin(),
    );
    let dir = hit.normalize();
    direction_to_yaw_pitch(&dir, &MONO_CAMERA_REST_POSITION)
}

/// Panorama-space (yaw, pitch) bounds, in radians, of the region a
/// [`CylindricalProjection`] actually has video painted on.
///
/// Sampled along the border of the source frame via
/// [`cylindrical_to_panorama`] rather than derived in closed form: the
/// video-rect-to-panorama mapping couples yaw and pitch (a corner and
/// an edge midpoint at the same `norm_x` land at different yaw once
/// `norm_y` shifts `y_world`), so the border is where the true extremes
/// occur and an axis-aligned bounding box over it is the same
/// conservative-envelope approach [`CoverageBoundary`] uses for the
/// L-shape path (there, tessellated across a 2D grid; here, cheap
/// enough to just sample every border pixel column/row).
///
/// A caller that wants to keep a virtual camera's *full viewport* (not
/// just its center ray) inside the painted region should inset these
/// bounds by half the viewport's angular extent on each side - see
/// [`crate::core::mono::MonoStitchCore`]'s pose-resolution step, which
/// does exactly that to keep the rendered crop free of the black
/// out-of-coverage wedges a pose too close to the edge produces.
pub fn cylindrical_panorama_bounds(config: &CylindricalProjectionConfig) -> ((f32, f32), (f32, f32)) {
    const SAMPLES: usize = 64;
    let mut yaw_min = f32::INFINITY;
    let mut yaw_max = f32::NEG_INFINITY;
    let mut pitch_min = f32::INFINITY;
    let mut pitch_max = f32::NEG_INFINITY;
    let mut visit = |norm_x: f32, norm_y: f32| {
        let pos = cylindrical_to_panorama(norm_x, norm_y, config);
        yaw_min = yaw_min.min(pos.yaw);
        yaw_max = yaw_max.max(pos.yaw);
        pitch_min = pitch_min.min(pos.pitch);
        pitch_max = pitch_max.max(pos.pitch);
    };
    for i in 0..=SAMPLES {
        let t = i as f32 / SAMPLES as f32;
        visit(t, 0.0);
        visit(t, 1.0);
        visit(0.0, t);
        visit(1.0, t);
    }
    ((yaw_min, yaw_max), (pitch_min, pitch_max))
}

/// Maximum Newton-Raphson iterations for KB4 inverse distortion.
const MAX_ITERATIONS: usize = 20;
/// Convergence threshold for Newton-Raphson.
const CONVERGENCE_EPS: f64 = 1e-10;

/// Map a detection in camera pixel space to the yaw/pitch needed to
/// center the virtual camera on it.
///
/// `norm_x` and `norm_y` are in normalized `[0.0, 1.0]` image coordinates
/// (as returned by [`Detection`](crate::detect::detector::Detection)).
///
/// Returns `None` if the inverse distortion fails to converge (rare,
/// indicates an extreme point far outside the valid lens area).
///
/// # Example
///
/// ```rust
/// use reco_core::projection::camera_to_panorama;
/// use reco_core::detect::detector::CameraId;
/// use reco_core::calibration::MatchCalibration;
/// use reco_core::render::scene::SceneGeometry;
///
/// # fn example(cal: &MatchCalibration) {
/// let aspect = cal.left.width as f32 / cal.left.height as f32;
/// let scene = SceneGeometry::from_layout_with_aspect(&cal.layout, aspect);
/// if let Some(pos) = camera_to_panorama(CameraId::Left, 0.5, 0.5, cal, &scene) {
///     println!("Center of left camera maps to yaw={:.3}, pitch={:.3}", pos.yaw, pos.pitch);
/// }
/// # }
/// ```
pub fn camera_to_panorama(
    camera: CameraId,
    norm_x: f32,
    norm_y: f32,
    calibration: &MatchCalibration,
    scene: &SceneGeometry,
) -> Option<ViewportPosition> {
    let params = match camera {
        CameraId::Left => &calibration.left,
        CameraId::Right => &calibration.right,
    };

    // Step 1: Inverse fisheye - camera pixel [0,1] -> plane UV (extended space)
    let plane_uv = inverse_fisheye(norm_x as f64, norm_y as f64, params)?;

    // Step 2: Plane UV -> 3D world point
    let world_point = plane_uv_to_world(plane_uv, camera, scene);

    // Step 3: World point -> yaw/pitch
    let dir = (world_point - Point3::from(Vector3::from(scene.camera_position))).normalize();
    Some(direction_to_yaw_pitch(&dir, &scene.camera_position))
}

/// Map a panorama position (yaw/pitch) back to a camera pixel coordinate.
///
/// This is the inverse of [`camera_to_panorama`]. Given a position in the
/// panoramic view, returns the corresponding normalized pixel coordinate
/// in the specified camera's image (or `None` if the position is outside
/// that camera's field of view).
///
/// Useful for:
/// - Projecting panorama-space detections back to camera images
/// - Computing panorama-to-pitch coordinate transforms (consumer territory)
/// - Overlay placement at specific panorama positions
pub fn panorama_to_camera(
    yaw: f32,
    pitch: f32,
    camera: CameraId,
    calibration: &MatchCalibration,
    scene: &SceneGeometry,
) -> Option<(f32, f32)> {
    use nalgebra::{Point3, Vector3};

    let params = match camera {
        CameraId::Left => &calibration.left,
        CameraId::Right => &calibration.right,
    };

    // Step 1: yaw/pitch -> world ray direction, through the same
    // VirtualCamera basis camera_to_panorama uses. Step 2 replaced
    // the previous naive "+Z forward" formula that broke the
    // Forward A vs Forward C roundtrip.
    let cam = VirtualCamera::new(&scene.camera_position);
    let dir = cam.yaw_pitch_to_direction(yaw, pitch);
    let cam_pos = Point3::from(cam.eye);

    // Step 2: ray-plane intersection.
    let model = match camera {
        CameraId::Left => scene.model_matrix_left(),
        CameraId::Right => scene.model_matrix_right(),
    };
    let plane_origin = model.transform_point(&Point3::new(0.0, 0.0, 0.0));
    let plane_normal = model
        .transform_vector(&Vector3::new(0.0, 0.0, 1.0))
        .normalize();

    let denom = plane_normal.dot(&dir);
    if denom.abs() < 1e-6 {
        return None; // Ray parallel to plane
    }
    let t = (plane_origin - cam_pos).dot(&plane_normal) / denom;
    if t <= 0.0 {
        return None; // Behind camera
    }
    let hit = cam_pos + dir * t;

    // Step 3: world hit -> extended plane UV. Reject hits outside
    // the plane's renderable region (texture UV [0, 1], equivalently
    // extended UV [-0.5, 1.5] is the full valid range but the plane
    // only covers [0, 1] inside that).
    let (uv_x, uv_y) = world_to_plane_uv(hit, camera, scene)?;
    let tex_u = (uv_x + 0.5) * 0.5;
    let tex_v = (uv_y + 0.5) * 0.5;
    if !(0.0..=1.0).contains(&tex_u) || !(0.0..=1.0).contains(&tex_v) {
        return None;
    }

    // Step 4: extended plane UV -> distorted normalized pixel via
    // the forward KB4 model. The previous implementation passed
    // texture UV in [0, 1] to `lens::undistorted_to_distorted` which
    // expects pixel coordinates; that blew up the lens math and
    // filtered almost every in-coverage point back out as None.
    let (norm_x, norm_y) = forward_fisheye(uv_x, uv_y, params);
    if (0.0..=1.0).contains(&norm_x) && (0.0..=1.0).contains(&norm_y) {
        Some((norm_x as f32, norm_y as f32))
    } else {
        None
    }
}

/// Compute the valid yaw/pitch bounds for a given FOV where no black
/// edges appear in the viewport.
///
/// Samples the visible edges of both camera planes and returns the
/// tightest bounds that keep the viewport fully within the projected
/// image area. Use this to clamp director output for "no-black" panning.
///
/// `aspect` is the viewport width/height ratio (e.g. 16/9 for 1080p).
///
/// Returns `(min_yaw, max_yaw, min_pitch, max_pitch)` in radians.
pub fn viewport_bounds(
    fov_degrees: f32,
    calibration: &MatchCalibration,
    scene: &SceneGeometry,
    aspect: f32,
) -> ViewportBounds {
    // fov_degrees is the VERTICAL FOV (nalgebra Perspective3 convention).
    // Derive horizontal FOV from aspect ratio using rectilinear projection.
    let half_vfov = (fov_degrees * 0.5).to_radians();
    let half_hfov = (half_vfov.tan() * aspect).atan();

    // The viewport corners reach further than edge midpoints due to the
    // tangent projection. At a corner, the angular distance from center
    // is: atan(sqrt(tan²(half_hfov) + tan²(half_vfov))). We account for
    // this by using the DIAGONAL angular extent for constraints, ensuring
    // even the corners stay inside coverage.
    //
    // For corner-aware bounds: when constraining yaw from a pitch bin,
    // the viewport extends half_hfov in yaw at the CENTER pitch, but at
    // the TOP/BOTTOM pitch (±half_vfov from center), the corner extends
    // even further. For a perspective projection, the corner yaw extent
    // at pitch offset dy is: atan(tan(half_hfov) / cos(dy)).
    // This is ~3-5% wider than half_hfov at the edges.
    let corner_hfov = (half_hfov.tan() / half_vfov.cos()).atan();
    let corner_vfov = (half_vfov.tan() / half_hfov.cos()).atan();

    // Sample the edges of both camera frames to find the coverage
    // boundary ("frontier") in panorama space. Using 2%/98% avoids
    // extreme fisheye corners where inverse distortion may diverge.
    let edge_steps: u32 = 40;
    let lo = 0.02_f32;
    let hi = 0.98_f32;
    let mut frontier: Vec<(f32, f32)> = Vec::with_capacity((edge_steps as usize + 1) * 8);

    for &camera in &[CameraId::Left, CameraId::Right] {
        for i in 0..=edge_steps {
            let t = lo + (hi - lo) * (i as f32 / edge_steps as f32);
            for &(nx, ny) in &[(lo, t), (hi, t), (t, lo), (t, hi)] {
                if let Some(pos) = camera_to_panorama(camera, nx, ny, calibration, scene) {
                    frontier.push((pos.yaw, pos.pitch));
                }
            }
        }
    }

    if frontier.is_empty() {
        return ViewportBounds {
            min_yaw: 0.0,
            max_yaw: 0.0,
            min_pitch: 0.0,
            max_pitch: 0.0,
        };
    }

    let pitch_min = frontier.iter().map(|p| p.1).fold(f32::MAX, f32::min);
    let pitch_max = frontier.iter().map(|p| p.1).fold(f32::MIN, f32::max);
    let yaw_min = frontier.iter().map(|p| p.0).fold(f32::MAX, f32::min);
    let yaw_max = frontier.iter().map(|p| p.0).fold(f32::MIN, f32::max);

    // Bin frontier points by pitch to find yaw coverage at each level.
    // Use corner_hfov (not half_hfov) so the viewport CORNERS stay
    // inside coverage, not just the edge midpoints.
    let n_bins: usize = 20;
    let pitch_range = pitch_max - pitch_min;
    let pitch_bin_size = pitch_range / n_bins as f32;
    let min_points_per_bin: usize = 4;

    let mut bound_min_yaw = f32::MIN;
    let mut bound_max_yaw = f32::MAX;

    for bin in 0..n_bins {
        let bin_lo = pitch_min + bin as f32 * pitch_bin_size;
        let bin_hi = bin_lo + pitch_bin_size;

        let (mut yaw_lo, mut yaw_hi, mut count) = (f32::MAX, f32::MIN, 0usize);
        for &(yaw, pitch) in &frontier {
            if pitch >= bin_lo && pitch < bin_hi {
                yaw_lo = yaw_lo.min(yaw);
                yaw_hi = yaw_hi.max(yaw);
                count += 1;
            }
        }

        if count < min_points_per_bin {
            continue;
        }

        bound_min_yaw = bound_min_yaw.max(yaw_lo + corner_hfov);
        bound_max_yaw = bound_max_yaw.min(yaw_hi - corner_hfov);
    }

    // Bin by yaw to find pitch coverage at each level.
    let yaw_range = yaw_max - yaw_min;
    let yaw_bin_size = yaw_range / n_bins as f32;

    let mut bound_min_pitch = f32::MIN;
    let mut bound_max_pitch = f32::MAX;

    for bin in 0..n_bins {
        let bin_lo = yaw_min + bin as f32 * yaw_bin_size;
        let bin_hi = bin_lo + yaw_bin_size;

        let (mut p_lo, mut p_hi, mut count) = (f32::MAX, f32::MIN, 0usize);
        for &(yaw, pitch) in &frontier {
            if yaw >= bin_lo && yaw < bin_hi {
                p_lo = p_lo.min(pitch);
                p_hi = p_hi.max(pitch);
                count += 1;
            }
        }

        if count < min_points_per_bin {
            continue;
        }

        bound_min_pitch = bound_min_pitch.max(p_lo + corner_vfov);
        bound_max_pitch = bound_max_pitch.min(p_hi - corner_vfov);
    }

    // Fallback if binning produced no constraints.
    if bound_min_yaw == f32::MIN {
        bound_min_yaw = yaw_min + corner_hfov;
    }
    if bound_max_yaw == f32::MAX {
        bound_max_yaw = yaw_max - corner_hfov;
    }
    if bound_min_pitch == f32::MIN {
        bound_min_pitch = pitch_min + corner_vfov;
    }
    if bound_max_pitch == f32::MAX {
        bound_max_pitch = pitch_max - corner_vfov;
    }

    // Collapse to midpoint if bounds inverted (coverage too narrow).
    if bound_min_yaw > bound_max_yaw {
        let mid = (bound_min_yaw + bound_max_yaw) * 0.5;
        bound_min_yaw = mid;
        bound_max_yaw = mid;
    }
    if bound_min_pitch > bound_max_pitch {
        let mid = (bound_min_pitch + bound_max_pitch) * 0.5;
        bound_min_pitch = mid;
        bound_max_pitch = mid;
    }

    ViewportBounds {
        min_yaw: bound_min_yaw,
        max_yaw: bound_max_yaw,
        min_pitch: bound_min_pitch,
        max_pitch: bound_max_pitch,
    }
}

/// Valid viewport bounds for "no-black" panning.
///
/// Clamp the director's yaw/pitch to these ranges to ensure the
/// viewport never shows black edges from the L-shaped projection.
#[derive(Debug, Clone, Copy)]
pub struct ViewportBounds {
    /// Minimum yaw in radians (leftmost pan).
    pub min_yaw: f32,
    /// Maximum yaw in radians (rightmost pan).
    pub max_yaw: f32,
    /// Minimum pitch in radians (lowest tilt).
    pub min_pitch: f32,
    /// Maximum pitch in radians (highest tilt).
    pub max_pitch: f32,
}

impl ViewportBounds {
    /// Clamp a viewport position to stay within these bounds.
    pub fn clamp(&self, position: ViewportPosition) -> ViewportPosition {
        ViewportPosition {
            yaw: position.yaw.clamp(self.min_yaw, self.max_yaw),
            pitch: position.pitch.clamp(self.min_pitch, self.max_pitch),
            fov_degrees: position.fov_degrees,
        }
    }
}

// ---- Internal functions ----

/// Forward KB4 fisheye: undistorted plane UV -> distorted camera pixel [0,1].
///
/// Mirror of [`inverse_fisheye`] in the same normalized-intrinsic
/// convention and the same extended-UV plane space (the shader's
/// `uv * 2.0 - 0.5` remap output). The polynomial delegates to
/// `reco_core::lens::kb4`, same canonical source as the
/// Newton-Raphson step in [`inverse_fisheye`].
fn forward_fisheye(uv_x: f64, uv_y: f64, params: &CameraParams) -> (f64, f64) {
    let w = params.width as f64;
    let h = params.height as f64;
    let fx = params.fx / w;
    let fy = params.fy / h;
    let cx = params.cx / w;
    let cy = params.cy / h;

    let x = (uv_x - cx) / fx;
    let y = (uv_y - cy) / fy;
    let r = (x * x + y * y).sqrt();

    if r < 1e-12 {
        return (cx, cy);
    }

    let scale = crate::lens::kb4::kb4_forward_scale(r, &params.d);
    (fx * x * scale + cx, fy * y * scale + cy)
}

/// Inverse KB4 fisheye: distorted camera pixel [0,1] -> undistorted plane UV.
///
/// Inverts the forward KB4 model used in the shader:
/// ```text
/// theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
/// ```
/// Uses Newton-Raphson to solve for theta given theta_d.
fn inverse_fisheye(dist_x: f64, dist_y: f64, params: &CameraParams) -> Option<(f64, f64)> {
    let w = params.width as f64;
    let h = params.height as f64;
    let fx = params.fx / w;
    let fy = params.fy / h;
    let cx = params.cx / w;
    let cy = params.cy / h;
    let k = params.d;

    // Normalized distorted coordinates
    let dx = (dist_x - cx) / fx;
    let dy = (dist_y - cy) / fy;
    let theta_d = (dx * dx + dy * dy).sqrt();

    if theta_d < 1e-12 {
        // At the optical center - no distortion
        return Some((cx, cy));
    }

    // Newton-Raphson: solve f(theta) = theta_d_poly(theta) - theta_d = 0, where
    // theta_d_poly lives in `reco_core::lens::kb4` (SYNC_WITH WGSL).
    let mut theta = theta_d; // initial guess
    for _ in 0..MAX_ITERATIONS {
        let f = crate::lens::kb4::theta_d(theta, &k) - theta_d;
        let f_prime = crate::lens::kb4::theta_d_prime(theta, &k);

        if f_prime.abs() < 1e-15 {
            return None; // degenerate
        }

        let delta = f / f_prime;
        theta -= delta;

        if delta.abs() < CONVERGENCE_EPS {
            break;
        }
    }

    // Recover undistorted coordinates
    let r = theta.tan(); // theta = atan(r) -> r = tan(theta)
    let scale = if theta.abs() < 1e-12 {
        1.0
    } else {
        theta_d / r
    };

    // Guard against Inf/NaN from degenerate theta (e.g. theta near pi/2
    // where tan diverges, or numerical edge cases).
    if !scale.is_finite() {
        return None;
    }

    let x = dx / scale;
    let y = dy / scale;

    // Plane UV in the extended [-0.5, 1.5] space used by the shader
    let uv_x = fx * x + cx;
    let uv_y = fy * y + cy;

    Some((uv_x, uv_y))
}

/// Map a detection in a raw KB4 fisheye camera's pixel space to the
/// yaw/pitch needed to center the mono virtual camera on it - the KB4
/// counterpart of [`cylindrical_to_panorama`] (replaced by this
/// function; the cylindrical model was the wrong one for raw,
/// unstitched camera footage - see `kb4_mono.wgsl`'s module doc).
///
/// Inverts the same forward KB4 model [`project_world_point`] and
/// `kb4_mono.wgsl`'s `fs_kb4_mono` use, via the same Newton-Raphson
/// solve [`inverse_fisheye`] uses - but reconstructs a full 3D unit
/// ray from `theta` directly (`sin`/`cos`) instead of `inverse_fisheye`'s
/// `r = tan(theta)` plane-space form, which is undefined/singular at
/// theta approaching or exceeding 90 degrees; this mono path's
/// self-calibration explicitly searches lenses up to 110 degrees
/// half-FOV, so that limitation would be reached in practice.
///
/// `norm_x`/`norm_y` are normalized `[0.0, 1.0]` image coordinates, in
/// the standard image convention (`[`Detection`]`'s convention, +Y
/// down from the top). Returns `None` if Newton-Raphson fails to
/// converge (mirrors [`inverse_fisheye`]).
pub fn kb4_mono_to_panorama(
    norm_x: f32,
    norm_y: f32,
    camera: &CameraParams,
) -> Option<ViewportPosition> {
    let (w, h) = (camera.width as f64, camera.height as f64);
    let fx = camera.fx / w;
    let fy = camera.fy / h;
    let cx = camera.cx / w;
    let cy = camera.cy / h;
    let k = camera.d;

    let dx = (norm_x as f64 - cx) / fx;
    let dy = (norm_y as f64 - cy) / fy;
    let theta_d = (dx * dx + dy * dy).sqrt();

    let dir_cv = if theta_d < 1e-12 {
        Vector3::new(0.0, 0.0, 1.0)
    } else {
        let mut theta = theta_d;
        for _ in 0..MAX_ITERATIONS {
            let f = crate::lens::kb4::theta_d(theta, &k) - theta_d;
            let f_prime = crate::lens::kb4::theta_d_prime(theta, &k);
            if f_prime.abs() < 1e-15 {
                return None;
            }
            let delta = f / f_prime;
            theta -= delta;
            if delta.abs() < CONVERGENCE_EPS {
                break;
            }
        }
        if !theta.is_finite() {
            return None;
        }
        let azimuth = dy.atan2(dx);
        Vector3::new(
            theta.sin() * azimuth.cos(),
            theta.sin() * azimuth.sin(),
            theta.cos(),
        )
    };

    // Bridge CV convention (+Y down) to this codebase's graphics
    // convention (+Y up, same as view_matrix/yaw/pitch everywhere
    // else) - the exact inverse of the negation `kb4_mono.wgsl`'s
    // fragment shader applies going the other direction.
    let dir_graphics = Vector3::new(dir_cv.x as f32, -dir_cv.y as f32, dir_cv.z as f32);
    Some(direction_to_yaw_pitch(&dir_graphics, &MONO_CAMERA_REST_POSITION))
}

/// Panorama-space (yaw, pitch) bounds, in radians, of the region a
/// mono KB4 camera actually has video for - the KB4 counterpart of
/// [`cylindrical_panorama_bounds`] (see its doc comment for why
/// border-sampling, not a closed form, is the right approach here
/// too: the pixel-to-panorama mapping isn't axis-separable).
///
/// Samples the source image's border; a border point whose distorted
/// radius exceeds `max_theta_rad`'s (via the forward KB4 polynomial)
/// is skipped rather than trusted - the same guard
/// `kb4_mono.wgsl`'s fragment shader applies, kept consistent here so
/// the pose clamp this feeds never promises coverage the renderer
/// would reject.
pub fn kb4_mono_panorama_bounds(
    camera: &CameraParams,
    max_theta_rad: f32,
) -> ((f32, f32), (f32, f32)) {
    const SAMPLES: usize = 64;
    let max_r_d = crate::lens::kb4::theta_d(max_theta_rad as f64, &camera.d);

    let mut yaw_min = f32::INFINITY;
    let mut yaw_max = f32::NEG_INFINITY;
    let mut pitch_min = f32::INFINITY;
    let mut pitch_max = f32::NEG_INFINITY;
    let mut visit = |norm_x: f32, norm_y: f32| {
        let (w, h) = (camera.width as f64, camera.height as f64);
        let dx = (norm_x as f64 - camera.cx / w) / (camera.fx / w);
        let dy = (norm_y as f64 - camera.cy / h) / (camera.fy / h);
        if (dx * dx + dy * dy).sqrt() > max_r_d {
            return;
        }
        if let Some(pos) = kb4_mono_to_panorama(norm_x, norm_y, camera) {
            yaw_min = yaw_min.min(pos.yaw);
            yaw_max = yaw_max.max(pos.yaw);
            pitch_min = pitch_min.min(pos.pitch);
            pitch_max = pitch_max.max(pos.pitch);
        }
    };
    for i in 0..=SAMPLES {
        let t = i as f32 / SAMPLES as f32;
        visit(t, 0.0);
        visit(t, 1.0);
        visit(0.0, t);
        visit(1.0, t);
    }

    // Every sample can fail - e.g. a badly-converged calibration whose
    // `k` coefficients make `theta_d` non-monotonic (even negative) at
    // `max_theta_rad`, which makes the `max_r_d` gate above reject
    // every single border point. Left as the initial +-infinity
    // sentinels, the caller's "inset by a margin, collapse to midpoint
    // if inverted" logic computes `infinity + -infinity = NaN` and
    // corrupts every downstream pose clamp. Degrade to a single safe
    // point (dead ahead) instead of propagating NaN - a bad
    // calibration should visibly under-perform, not crash the run.
    if !yaw_min.is_finite() || !yaw_max.is_finite() || !pitch_min.is_finite() || !pitch_max.is_finite()
    {
        log::error!(
            "kb4_mono_panorama_bounds: no source-image border sample produced a valid \
             panorama position (camera d={:?}, max_theta_rad={max_theta_rad}) - the \
             calibration is likely degenerate (e.g. distortion coefficients pinned at a \
             solver bound). Falling back to a single dead-ahead point instead of crashing; \
             re-run calibrate-mono.",
            camera.d
        );
        return ((0.0, 0.0), (0.0, 0.0));
    }
    ((yaw_min, yaw_max), (pitch_min, pitch_max))
}

/// Project a 3D world point through a posed KB4 fisheye camera to a
/// normalized `[0,1]` pixel coordinate.
///
/// Standard Kannala-Brandt equidistant-fisheye projection: `pose`
/// transforms world -> camera-local coordinates (camera looks down
/// `+Z`, matching every other camera convention in this module -
/// see [`MONO_CAMERA_REST_POSITION`]'s doc comment). Works
/// directly on the ray's `(x, y, z)` via `atan2`, unlike
/// [`forward_fisheye`]/[`kb4_forward_scale`](crate::lens::kb4_forward_scale)'s
/// `r = tan(theta)` plane-UV convention, which is only valid for
/// `theta < FRAC_PI_2` - a wide fisheye's field of view routinely
/// exceeds that, so this function is the one to use for a camera's
/// *own* raw field of view (mono self-calibration) rather than a
/// flat stitching-plane's UV space.
///
/// Returns `None` for points effectively at the camera (degenerate
/// direction) or beyond `max_theta_rad` from the optical axis (default
/// caller should pass `f64::consts::PI` for "no limit" during
/// calibration solves, and a calibrated lens's actual max field of
/// view once known).
pub fn project_world_point(
    pose: &nalgebra::Isometry3<f64>,
    world_point: &nalgebra::Point3<f64>,
    params: &CameraParams,
    max_theta_rad: f64,
) -> Option<(f64, f64)> {
    let p = pose.transform_point(world_point);
    let r = (p.x * p.x + p.y * p.y).sqrt();
    let theta = r.atan2(p.z);
    if !(0.0..=max_theta_rad).contains(&theta) {
        return None;
    }
    let theta_d = crate::lens::kb4::theta_d(theta, &params.d);
    let (sx, sy) = if r < 1e-12 {
        (0.0, 0.0)
    } else {
        (theta_d * p.x / r, theta_d * p.y / r)
    };
    let px = params.fx * sx + params.cx;
    let py = params.fy * sy + params.cy;
    Some((px / params.width as f64, py / params.height as f64))
}

/// Exact inverse of [`plane_uv_to_world`].
///
/// Given a world-space point that lies on the named camera's plane,
/// returns its extended-UV coordinate (shader space `[-0.5, 1.5]`).
/// Off-plane points project via the model-matrix inverse: the z
/// component of the model-local position is discarded, so the result
/// is the orthographic projection onto the plane, NOT the ray-plane
/// intersection that `panorama_to_camera` does as a prior step.
fn world_to_plane_uv(
    world: nalgebra::Point3<f32>,
    camera: CameraId,
    scene: &SceneGeometry,
) -> Option<(f64, f64)> {
    let model = match camera {
        CameraId::Left => scene.model_matrix_left(),
        CameraId::Right => scene.model_matrix_right(),
    };
    let inv_model = model.try_inverse()?;
    let local = inv_model.transform_point(&world);

    // local -> texture UV [0,1] (inverse of plane_uv_to_world's inner
    // texture->local step, with plane_width = 1.0 baked in).
    let tex_u = local.x / scene.plane_width + 0.5;
    let tex_v = 0.5 - local.y * scene.plane_aspect / scene.plane_width;

    // Texture UV -> extended shader UV (inverse of `uv * 2.0 - 0.5`).
    let uv_x = (tex_u * 2.0 - 0.5) as f64;
    let uv_y = (tex_v * 2.0 - 0.5) as f64;
    Some((uv_x, uv_y))
}

/// Convert a plane UV (in extended shader space) to a 3D world point.
fn plane_uv_to_world(uv: (f64, f64), camera: CameraId, scene: &SceneGeometry) -> Point3<f32> {
    // Extended UV -> texture UV [0,1]
    let tex_u = ((uv.0 + 0.5) / 2.0) as f32;
    let tex_v = ((uv.1 + 0.5) / 2.0) as f32;

    // Texture UV -> local quad position (matches quad_vertices)
    let local_x = tex_u - 0.5;
    let local_y = (0.5 - tex_v) / scene.plane_aspect;

    let local_point = nalgebra::Vector4::new(local_x, local_y, 0.0, 1.0);
    let model = match camera {
        CameraId::Left => scene.model_matrix_left(),
        CameraId::Right => scene.model_matrix_right(),
    };

    let world = model * local_point;
    Point3::new(world.x, world.y, world.z)
}

/// Decompose a direction vector into yaw/pitch relative to the virtual camera.
///
/// Thin wrapper over [`VirtualCamera::direction_to_yaw_pitch`] kept
/// for the existing call sites until they carry a `VirtualCamera`
/// directly. Panners, directors, and the render loop all share the
/// same basis through this path.
pub(crate) fn direction_to_yaw_pitch(
    dir: &Vector3<f32>,
    camera_position: &[f32; 3],
) -> ViewportPosition {
    VirtualCamera::new(camera_position).direction_to_yaw_pitch(dir)
}

/// Exact inverse of [`direction_to_yaw_pitch`]. Only called from
/// tests today; production panorama_to_camera uses the method on
/// [`VirtualCamera`] directly.
#[cfg(test)]
pub(crate) fn yaw_pitch_to_direction(
    yaw: f32,
    pitch: f32,
    camera_position: &[f32; 3],
) -> Vector3<f32> {
    VirtualCamera::new(camera_position).yaw_pitch_to_direction(yaw, pitch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::{CameraParams, MatchCalibration, PlaneLayout};

    fn test_scene(cal: &MatchCalibration) -> SceneGeometry {
        let aspect = cal.left.width as f32 / cal.left.height as f32;
        SceneGeometry::from_layout_with_aspect(&cal.layout, aspect)
    }

    fn test_calibration() -> MatchCalibration {
        MatchCalibration {
            left: CameraParams {
                width: 3840,
                height: 2160,
                fx: 1796.32,
                fy: 1797.22,
                cx: 1919.37,
                cy: 1063.17,
                d: [0.0342, 0.0677, -0.0741, 0.0299],
            },
            right: CameraParams {
                width: 3840,
                height: 2160,
                fx: 1796.32,
                fy: 1797.22,
                cx: 1919.37,
                cy: 1063.17,
                d: [0.0342, 0.0677, -0.0741, 0.0299],
            },
            layout: PlaneLayout {
                camera_axis_offset: 0.2398,
                intersect: 0.5446,
                x_ty: 0.00476,
                x_rz: 0.00753,
                z_rx: -0.00431,
                x_rx: 0.0,
                z_rz: 0.0,
            },
            rig_tilt: 0.0,
            rig_roll: 0.0,
            sync_offset: 0,
            field_roi: None,
            lens_correction_amount: 1.0,
            blend_width: 0.05,
        }
    }

    #[test]
    fn optical_center_maps_to_known_position() {
        let cal = test_calibration();
        let scene = test_scene(&cal);

        // Optical center of the left camera (cx/w, cy/h)
        let cx = cal.left.cx as f32 / cal.left.width as f32;
        let cy = cal.left.cy as f32 / cal.left.height as f32;

        let pos = camera_to_panorama(CameraId::Left, cx, cy, &cal, &scene);
        assert!(pos.is_some(), "optical center should map successfully");
        let pos = pos.unwrap();
        // The optical center should produce a valid yaw/pitch (no NaN)
        assert!(pos.yaw.is_finite(), "yaw should be finite");
        assert!(pos.pitch.is_finite(), "pitch should be finite");
    }

    #[test]
    fn left_camera_left_edge_yaw_differs_from_center() {
        let cal = test_calibration();
        let scene = test_scene(&cal);

        let center = camera_to_panorama(CameraId::Left, 0.5, 0.5, &cal, &scene).unwrap();
        let left_edge = camera_to_panorama(CameraId::Left, 0.1, 0.5, &cal, &scene).unwrap();

        // The left edge of the left camera image maps to a different
        // part of the panorama than the center; this test just
        // asserts the pipeline is position-sensitive and doesn't
        // collapse distinct image positions to the same yaw. Sign
        // conventions are validated end-to-end by visual review,
        // not here.
        assert!(
            (left_edge.yaw - center.yaw).abs() > 0.1,
            "left edge yaw ({:.4}) and center yaw ({:.4}) should differ by > 0.1 rad",
            left_edge.yaw,
            center.yaw
        );
    }

    #[test]
    fn right_camera_produces_different_yaw_than_left() {
        let cal = test_calibration();
        let scene = test_scene(&cal);

        let left_center = camera_to_panorama(CameraId::Left, 0.5, 0.5, &cal, &scene).unwrap();
        let right_center = camera_to_panorama(CameraId::Right, 0.5, 0.5, &cal, &scene).unwrap();

        // The two cameras face different directions, so their centers
        // should map to different yaw values
        assert!(
            (left_center.yaw - right_center.yaw).abs() > 0.01,
            "left ({:.4}) and right ({:.4}) camera centers should differ in yaw",
            left_center.yaw,
            right_center.yaw
        );
    }

    #[test]
    fn yaw_pitch_to_direction_roundtrips_with_direction_to_yaw_pitch() {
        // Step 1a: the two helpers must form an exact bijection on the
        // (yaw, pitch) grid used by panners and directors. All
        // shipping scenes set `camera_position = [d, 0, d]` (see
        // SceneGeometry::from_layout_with_aspect), so eye.y = 0 is
        // the real invariant; test positions honor that. Pitch stays
        // clear of +-pi/2 where yaw is undefined.
        let camera_positions: [[f32; 3]; 3] = [[0.24, 0.0, 0.24], [0.3, 0.0, 0.2], [0.1, 0.0, 0.5]];

        let yaw_steps = [-1.2_f32, -0.6, -0.2, 0.0, 0.2, 0.6, 1.2];
        let pitch_steps = [-0.9_f32, -0.4, -0.1, 0.0, 0.1, 0.4, 0.9];

        for cam in &camera_positions {
            for &yaw in &yaw_steps {
                for &pitch in &pitch_steps {
                    let dir = yaw_pitch_to_direction(yaw, pitch, cam);
                    let norm = dir.norm();
                    assert!(
                        (norm - 1.0).abs() < 1e-5,
                        "direction must be unit, got |dir| = {norm} for cam={cam:?} yaw={yaw} pitch={pitch}"
                    );

                    let pos = direction_to_yaw_pitch(&dir, cam);
                    assert!(
                        (pos.yaw - yaw).abs() < 1e-4,
                        "yaw mismatch for cam={cam:?}: sent {yaw}, got {} (dir={dir:?})",
                        pos.yaw
                    );
                    assert!(
                        (pos.pitch - pitch).abs() < 1e-4,
                        "pitch mismatch for cam={cam:?}: sent {pitch}, got {} (dir={dir:?})",
                        pos.pitch
                    );
                }
            }
        }
    }

    #[test]
    fn world_to_plane_uv_roundtrips_with_plane_uv_to_world() {
        // Step 1b: extended UV -> world -> extended UV must be the
        // identity. Covers both camera planes and a grid spanning the
        // shader's extended range [-0.5, 1.5] (including points
        // outside the [0,1] texture box so we catch any implicit
        // clamp).
        let cal = test_calibration();
        let scene = test_scene(&cal);

        let uv_steps = [-0.3_f64, -0.1, 0.0, 0.25, 0.5, 0.75, 1.0, 1.1, 1.4];

        for &camera in &[CameraId::Left, CameraId::Right] {
            for &u in &uv_steps {
                for &v in &uv_steps {
                    let world = plane_uv_to_world((u, v), camera, &scene);
                    let back = world_to_plane_uv(world, camera, &scene)
                        .expect("model matrix should be invertible");

                    assert!(
                        (back.0 - u).abs() < 1e-5,
                        "uv.x mismatch for camera={camera:?}: sent {u}, got {} (world={world:?})",
                        back.0
                    );
                    assert!(
                        (back.1 - v).abs() < 1e-5,
                        "uv.y mismatch for camera={camera:?}: sent {v}, got {} (world={world:?})",
                        back.1
                    );
                }
            }
        }
    }

    #[test]
    fn inverse_fisheye_roundtrips_with_forward_fisheye_on_pixel_grid() {
        // Step 1c: normalized distorted pixel -> extended plane UV ->
        // back to normalized distorted pixel must be the identity on
        // a 10x10 grid inside the valid image area. Realistic KB4
        // coefficients (the GoPro HERO10 4K test calibration) make
        // this representative of shipping workloads.
        let params = CameraParams {
            width: 3840,
            height: 2160,
            fx: 1796.32,
            fy: 1797.22,
            cx: 1919.37,
            cy: 1063.17,
            d: [0.0342, 0.0677, -0.0741, 0.0299],
        };

        let steps = 10;
        // Stay inside [0.1, 0.9] to avoid extreme fisheye corners where
        // Newton-Raphson may refuse to converge (documented in
        // `inverse_fisheye`'s None return).
        let lo = 0.1_f64;
        let hi = 0.9_f64;

        for ix in 0..=steps {
            for iy in 0..=steps {
                let nx = lo + (hi - lo) * (ix as f64 / steps as f64);
                let ny = lo + (hi - lo) * (iy as f64 / steps as f64);

                let plane_uv = inverse_fisheye(nx, ny, &params)
                    .expect("inverse_fisheye should converge inside valid area");
                let (back_x, back_y) = forward_fisheye(plane_uv.0, plane_uv.1, &params);

                assert!(
                    (back_x - nx).abs() < 1e-6,
                    "x mismatch at ({nx}, {ny}): got {back_x}, plane_uv={plane_uv:?}"
                );
                assert!(
                    (back_y - ny).abs() < 1e-6,
                    "y mismatch at ({nx}, {ny}): got {back_y}, plane_uv={plane_uv:?}"
                );
            }
        }
    }

    #[test]
    fn camera_to_panorama_roundtrips_with_panorama_to_camera() {
        // Step 1d (un-ignored by Step 2): the full forward chain
        // (camera_to_panorama) then backward chain (panorama_to_camera)
        // must return the same normalized pixel position for points
        // that lie unambiguously within one camera's coverage.
        //
        // Pre-Step-2 panorama_to_camera used a naive "+Z forward" ray
        // (breaking the Forward A vs Forward C agreement) AND passed
        // texture UV to a pixel-space lens helper (filtering
        // in-coverage results back out as None). Step 2 replaced
        // both with VirtualCamera + world_to_plane_uv + forward_fisheye.
        let cal = test_calibration();
        let scene = test_scene(&cal);

        let steps = 5;
        let lo = 0.3_f32;
        let hi = 0.7_f32;

        for &camera in &[CameraId::Left, CameraId::Right] {
            for ix in 0..=steps {
                for iy in 0..=steps {
                    let nx = lo + (hi - lo) * (ix as f32 / steps as f32);
                    let ny = lo + (hi - lo) * (iy as f32 / steps as f32);

                    let pos = camera_to_panorama(camera, nx, ny, &cal, &scene)
                        .expect("forward projection should succeed inside coverage");

                    let back = panorama_to_camera(pos.yaw, pos.pitch, camera, &cal, &scene)
                        .expect("backward projection should land on the same camera");

                    assert!(
                        (back.0 - nx).abs() < 1e-3,
                        "x mismatch for camera={camera:?}: sent {nx}, got {} (yaw={}, pitch={})",
                        back.0,
                        pos.yaw,
                        pos.pitch
                    );
                    assert!(
                        (back.1 - ny).abs() < 1e-3,
                        "y mismatch for camera={camera:?}: sent {ny}, got {} (yaw={}, pitch={})",
                        back.1,
                        pos.yaw,
                        pos.pitch
                    );
                }
            }
        }
    }

    #[test]
    fn project_world_point_on_axis_maps_to_principal_point() {
        let params = CameraParams {
            width: 3840,
            height: 2160,
            fx: 1796.32,
            fy: 1797.22,
            cx: 1919.37,
            cy: 1063.17,
            d: [0.0342, 0.0677, -0.0741, 0.0299],
        };
        let pose = nalgebra::Isometry3::identity();
        // Straight ahead on the optical axis (+Z), any distance.
        let world_point = nalgebra::Point3::new(0.0, 0.0, 5.0);
        let (px, py) =
            project_world_point(&pose, &world_point, &params, std::f64::consts::PI).unwrap();
        assert!((px - params.cx / params.width as f64).abs() < 1e-9);
        assert!((py - params.cy / params.height as f64).abs() < 1e-9);
    }

    #[test]
    fn project_world_point_zero_distortion_matches_pinhole() {
        let params = CameraParams {
            width: 1000,
            height: 1000,
            fx: 500.0,
            fy: 500.0,
            cx: 500.0,
            cy: 500.0,
            d: [0.0, 0.0, 0.0, 0.0],
        };
        let pose = nalgebra::Isometry3::identity();
        // A SMALL off-axis point (theta ~0.77 deg): the KB4-equidistant
        // model (theta_d = theta exactly, since d=0) and the standard
        // pinhole model (tan(theta)-based) are different formulas in
        // general, but converge for small theta - this is deliberately
        // not a wide-angle point (see the dedicated max_theta test for
        // that regime), just confirming the near-axis behavior is sane.
        let world_point = nalgebra::Point3::new(0.05, 0.02, 4.0);
        let (px, py) =
            project_world_point(&pose, &world_point, &params, std::f64::consts::PI).unwrap();

        let pinhole_x = params.fx * world_point.x / world_point.z + params.cx;
        let pinhole_y = params.fy * world_point.y / world_point.z + params.cy;
        assert!((px - pinhole_x / params.width as f64).abs() < 1e-5);
        assert!((py - pinhole_y / params.height as f64).abs() < 1e-5);
    }

    #[test]
    fn project_world_point_rejects_beyond_max_theta() {
        let params = CameraParams {
            width: 1000,
            height: 1000,
            fx: 500.0,
            fy: 500.0,
            cx: 500.0,
            cy: 500.0,
            d: [0.0, 0.0, 0.0, 0.0],
        };
        let pose = nalgebra::Isometry3::identity();
        // 90 degrees off-axis (in the camera's own local x-y plane).
        let world_point = nalgebra::Point3::new(1.0, 0.0, 0.0);
        assert!(project_world_point(&pose, &world_point, &params, 1.0).is_none());
        assert!(
            project_world_point(&pose, &world_point, &params, std::f64::consts::PI).is_some()
        );
    }

    #[test]
    fn kb4_mono_to_panorama_roundtrips_with_project_world_point() {
        // Validates both directions of the CV-Y-down <-> graphics-Y-up
        // bridge at once: forward-project a known ray via
        // `project_world_point` (pure CV convention, no bridging),
        // invert the resulting pixel via `kb4_mono_to_panorama` (which
        // DOES bridge), and check the result matches
        // `direction_to_yaw_pitch` applied directly to the same ray's
        // graphics-convention form (y negated).
        let camera = CameraParams {
            width: 1920,
            height: 1080,
            fx: 800.0,
            fy: 800.0,
            cx: 960.0,
            cy: 540.0,
            d: [0.02, -0.01, 0.0, 0.0],
        };
        let pose = nalgebra::Isometry3::identity();

        for &(theta_deg, azimuth_deg) in &[
            (0.0_f64, 0.0_f64),
            (20.0, 0.0),
            (20.0, 90.0),
            (30.0, 180.0),
            (15.0, 270.0),
        ] {
            let theta = theta_deg.to_radians();
            let az = azimuth_deg.to_radians();
            let dir_cv = (
                theta.sin() * az.cos(),
                theta.sin() * az.sin(),
                theta.cos(),
            );
            let world_point = Point3::new(dir_cv.0 * 5.0, dir_cv.1 * 5.0, dir_cv.2 * 5.0);
            let (px, py) = project_world_point(&pose, &world_point, &camera, std::f64::consts::PI)
                .expect("forward projection should succeed for a moderate angle");
            let pos = kb4_mono_to_panorama(px as f32, py as f32, &camera)
                .expect("inverse should converge");

            let dir_graphics = Vector3::new(dir_cv.0 as f32, -dir_cv.1 as f32, dir_cv.2 as f32);
            let expected = direction_to_yaw_pitch(&dir_graphics, &MONO_CAMERA_REST_POSITION);
            assert!(
                (pos.yaw - expected.yaw).abs() < 1e-3,
                "theta={theta_deg} az={azimuth_deg}: yaw {} != expected {}",
                pos.yaw,
                expected.yaw
            );
            assert!(
                (pos.pitch - expected.pitch).abs() < 1e-3,
                "theta={theta_deg} az={azimuth_deg}: pitch {} != expected {}",
                pos.pitch,
                expected.pitch
            );
        }
    }

    #[test]
    fn kb4_mono_panorama_bounds_are_symmetric_and_nonempty() {
        let camera = CameraParams {
            width: 1920,
            height: 1080,
            fx: 800.0,
            fy: 800.0,
            cx: 960.0,
            cy: 540.0,
            d: [0.0, 0.0, 0.0, 0.0],
        };
        let ((yaw_min, yaw_max), (pitch_min, pitch_max)) =
            kb4_mono_panorama_bounds(&camera, std::f32::consts::PI * 0.75);

        assert!(yaw_min < 0.0 && yaw_max > 0.0, "yaw range must straddle center");
        assert!((yaw_min + yaw_max).abs() < 1e-2, "yaw bounds roughly symmetric");
        assert!(pitch_min < 0.0 && pitch_max > 0.0);
        assert!((pitch_min + pitch_max).abs() < 1e-2);
    }

    #[test]
    fn kb4_mono_panorama_bounds_shrink_with_tighter_max_theta() {
        let camera = CameraParams {
            width: 1920,
            height: 1080,
            fx: 800.0,
            fy: 800.0,
            cx: 960.0,
            cy: 540.0,
            d: [0.0, 0.0, 0.0, 0.0],
        };
        let (wide_yaw, _) = kb4_mono_panorama_bounds(&camera, std::f32::consts::PI * 0.75);
        let (tight_yaw, _) = kb4_mono_panorama_bounds(&camera, 0.2);
        assert!(
            tight_yaw.1 - tight_yaw.0 < wide_yaw.1 - wide_yaw.0,
            "a tighter max_theta must not produce a wider or equal envelope"
        );
    }

    #[test]
    fn inverse_fisheye_roundtrip_at_center() {
        let params = CameraParams {
            width: 3840,
            height: 2160,
            fx: 1796.32,
            fy: 1797.22,
            cx: 1919.37,
            cy: 1063.17,
            d: [0.0342, 0.0677, -0.0741, 0.0299],
        };

        // At the optical center, distortion should be zero
        let cx = params.cx / params.width as f64;
        let cy = params.cy / params.height as f64;
        let result = inverse_fisheye(cx, cy, &params).unwrap();
        assert!(
            (result.0 - cx).abs() < 1e-6 && (result.1 - cy).abs() < 1e-6,
            "optical center should be a fixed point: got ({:.6}, {:.6}), expected ({:.6}, {:.6})",
            result.0,
            result.1,
            cx,
            cy
        );
    }

    #[test]
    fn viewport_bounds_are_valid() {
        let cal = test_calibration();
        let scene = test_scene(&cal);

        // Use a narrower FOV to ensure bounds are valid
        let bounds = viewport_bounds(40.0, &cal, &scene, 16.0 / 9.0);
        assert!(
            bounds.min_yaw < bounds.max_yaw,
            "yaw range should be valid: {:.4}..{:.4}",
            bounds.min_yaw,
            bounds.max_yaw
        );
        assert!(
            bounds.min_pitch < bounds.max_pitch,
            "pitch range should be valid: {:.4}..{:.4}",
            bounds.min_pitch,
            bounds.max_pitch
        );
        // With 40 deg FOV, the valid range should be non-trivial
        assert!(
            bounds.max_yaw - bounds.min_yaw > 0.01,
            "yaw range too small: {:.4}..{:.4}",
            bounds.min_yaw,
            bounds.max_yaw
        );
    }

    #[test]
    fn wider_fov_produces_tighter_bounds() {
        let cal = test_calibration();
        let scene = test_scene(&cal);

        let narrow = viewport_bounds(30.0, &cal, &scene, 16.0 / 9.0);
        let wide = viewport_bounds(60.0, &cal, &scene, 16.0 / 9.0);

        // Wider FOV should produce tighter (or equal) yaw bounds
        assert!(
            wide.min_yaw >= narrow.min_yaw,
            "wider FOV min_yaw ({:.4}) should be >= narrow ({:.4})",
            wide.min_yaw,
            narrow.min_yaw
        );
        assert!(
            wide.max_yaw <= narrow.max_yaw,
            "wider FOV max_yaw ({:.4}) should be <= narrow ({:.4})",
            wide.max_yaw,
            narrow.max_yaw
        );
    }

    #[test]
    fn zero_distortion_produces_identity_mapping() {
        let params = CameraParams {
            width: 1920,
            height: 1080,
            fx: 960.0,
            fy: 540.0,
            cx: 960.0,
            cy: 540.0,
            d: [0.0, 0.0, 0.0, 0.0],
        };

        // With zero distortion and fx=width/2, cx=width/2, the mapping
        // should be close to identity
        let result = inverse_fisheye(0.5, 0.5, &params).unwrap();
        assert!(
            (result.0 - 0.5).abs() < 1e-6 && (result.1 - 0.5).abs() < 1e-6,
            "zero-distortion center should map to itself"
        );
    }

    // --- point_in_polygon tests ---

    /// Unit square: [0,0] -> [1,0] -> [1,1] -> [0,1].
    fn unit_square() -> Vec<[f64; 2]> {
        vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    }

    #[test]
    fn pip_center_of_square() {
        assert!(point_in_polygon([0.5, 0.5], &unit_square()));
    }

    #[test]
    fn pip_outside_square() {
        assert!(!point_in_polygon([1.5, 0.5], &unit_square()));
        assert!(!point_in_polygon([-0.1, 0.5], &unit_square()));
        assert!(!point_in_polygon([0.5, -0.1], &unit_square()));
        assert!(!point_in_polygon([0.5, 1.1], &unit_square()));
    }

    #[test]
    fn pip_triangle() {
        let triangle = vec![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]];
        // Inside
        assert!(point_in_polygon([0.5, 0.3], &triangle));
        // Outside (right of the triangle)
        assert!(!point_in_polygon([0.9, 0.8], &triangle));
    }

    #[test]
    fn pip_concave_l_shape() {
        // L-shaped polygon (concave):
        //   (0,0) -> (1,0) -> (1,0.5) -> (0.5,0.5) -> (0.5,1) -> (0,1)
        let l_shape = vec![
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.5],
            [0.5, 0.5],
            [0.5, 1.0],
            [0.0, 1.0],
        ];
        // Inside the bottom-right arm
        assert!(point_in_polygon([0.75, 0.25], &l_shape));
        // Inside the top-left arm
        assert!(point_in_polygon([0.25, 0.75], &l_shape));
        // In the concave cutout (top-right) - should be outside
        assert!(!point_in_polygon([0.75, 0.75], &l_shape));
    }

    #[test]
    fn pip_degenerate_polygon() {
        // Fewer than 3 vertices: always false.
        assert!(!point_in_polygon([0.5, 0.5], &[]));
        assert!(!point_in_polygon([0.5, 0.5], &[[0.0, 0.0]]));
        assert!(!point_in_polygon([0.5, 0.5], &[[0.0, 0.0], [1.0, 1.0]]));
    }

    #[test]
    fn pip_near_edge_of_square() {
        // Just inside the edge
        assert!(point_in_polygon([0.001, 0.5], &unit_square()));
        assert!(point_in_polygon([0.999, 0.5], &unit_square()));
    }

    #[test]
    fn coverage_yaw_and_pitch_ranges_match_internal_state() {
        let cal = test_calibration();
        let scene = test_scene(&cal);
        let coverage = CoverageBoundary::from_calibration(&cal, &scene);

        let (yaw_min, yaw_max) = coverage.yaw_range();
        assert!(yaw_min < yaw_max, "yaw range must be non-empty");
        assert!(yaw_min.is_finite() && yaw_max.is_finite());

        let (pitch_min, pitch_max) = coverage.pitch_range();
        assert_eq!(pitch_min, coverage.pitch_min);
        assert_eq!(pitch_max, coverage.pitch_max);
    }

    #[test]
    fn coverage_yaw_range_is_widest_slice_envelope() {
        // yaw_range() must be at least as wide as any yaw_range_at(pitch)
        // sample, since it's the envelope over all pitch slices.
        let cal = test_calibration();
        let scene = test_scene(&cal);
        let coverage = CoverageBoundary::from_calibration(&cal, &scene);

        let (y_lo_global, y_hi_global) = coverage.yaw_range();
        let (p_lo, p_hi) = coverage.pitch_range();

        for i in 0..=10 {
            let t = i as f32 / 10.0;
            let pitch = p_lo + t * (p_hi - p_lo);
            let (y_lo, y_hi) = coverage.yaw_range_at(pitch);
            if y_lo > y_hi {
                continue; // degenerate interpolation outside coverage
            }
            assert!(
                y_lo >= y_lo_global - 1e-4,
                "pitch {pitch} yaw lo {y_lo} below global {y_lo_global}"
            );
            assert!(
                y_hi <= y_hi_global + 1e-4,
                "pitch {pitch} yaw hi {y_hi} above global {y_hi_global}"
            );
        }
    }

    #[test]
    fn panorama_extent_normalize_is_in_range_at_corners() {
        let ext = PanoramaExtent {
            yaw_min: -0.5,
            yaw_max: 0.5,
            pitch_min: -0.3,
            pitch_max: 0.3,
        };
        assert_eq!(ext.yaw_span(), 1.0);
        assert_eq!(ext.pitch_span(), 0.6);

        let (u, v) = ext.normalize(-0.5, -0.3).unwrap();
        assert!((u - 0.0).abs() < 1e-6);
        assert!((v - 0.0).abs() < 1e-6);

        let (u, v) = ext.normalize(0.5, 0.3).unwrap();
        assert!((u - 1.0).abs() < 1e-6);
        assert!((v - 1.0).abs() < 1e-6);

        let (u, v) = ext.normalize(0.0, 0.0).unwrap();
        assert!((u - 0.5).abs() < 1e-6);
        assert!((v - 0.5).abs() < 1e-6);
    }

    #[test]
    fn panorama_extent_normalize_rejects_degenerate() {
        let ext = PanoramaExtent {
            yaw_min: 0.0,
            yaw_max: 0.0,
            pitch_min: 0.0,
            pitch_max: 0.0,
        };
        assert!(ext.normalize(0.0, 0.0).is_none());
    }

    // -- B-30 NaN-resilience regression tests --

    #[test]
    fn safe_clamp_rejects_nan_yaw() {
        let cal = test_calibration();
        let scene = test_scene(&cal);
        let coverage = CoverageBoundary::from_calibration(&cal, &scene);
        let out = coverage.safe_clamp(f32::NAN, 0.0, 75.0, 16.0 / 9.0, 0.0);
        assert!(out.yaw.is_finite(), "yaw must be finite, got {}", out.yaw);
        assert!(
            out.pitch.is_finite(),
            "pitch must be finite, got {}",
            out.pitch
        );
    }

    #[test]
    fn safe_clamp_rejects_nan_pitch() {
        let cal = test_calibration();
        let scene = test_scene(&cal);
        let coverage = CoverageBoundary::from_calibration(&cal, &scene);
        let out = coverage.safe_clamp(0.0, f32::NAN, 75.0, 16.0 / 9.0, 0.0);
        assert!(out.yaw.is_finite());
        assert!(out.pitch.is_finite());
    }

    #[test]
    fn safe_clamp_rejects_nan_fov() {
        let cal = test_calibration();
        let scene = test_scene(&cal);
        let coverage = CoverageBoundary::from_calibration(&cal, &scene);
        let out = coverage.safe_clamp(0.0, 0.0, f32::NAN, 16.0 / 9.0, 0.0);
        assert!(out.yaw.is_finite());
        assert!(out.pitch.is_finite());
    }

    #[test]
    fn safe_clamp_rejects_infinite_inputs() {
        let cal = test_calibration();
        let scene = test_scene(&cal);
        let coverage = CoverageBoundary::from_calibration(&cal, &scene);
        let out = coverage.safe_clamp(f32::INFINITY, 0.0, 75.0, 16.0 / 9.0, 0.0);
        assert!(out.yaw.is_finite());
        assert!(out.pitch.is_finite());
        let out = coverage.safe_clamp(0.0, f32::NEG_INFINITY, 75.0, 16.0 / 9.0, 0.0);
        assert!(out.yaw.is_finite());
        assert!(out.pitch.is_finite());
    }

    // -- M3 foundation: Projection trait tests --

    #[test]
    fn l_shape_projection_identifies_itself() {
        let p = LShapeProjection;
        assert_eq!(p.name(), "l-shape-stereo-2camera");
        assert_eq!(p.camera_count(), 2);
    }

    #[test]
    fn projection_is_dyn_compatible() {
        // Core invariant: StitchCore will hold `Box<dyn Projection>`.
        // Verify the trait bounds allow that today and that Send+Sync
        // both hold.
        let projections: Vec<Box<dyn Projection>> = vec![Box::new(LShapeProjection)];
        fn assert_send_sync<T: Send + Sync + ?Sized>() {}
        assert_send_sync::<dyn Projection>();
        assert_eq!(projections[0].camera_count(), 2);
    }

    #[test]
    fn l_shape_projection_wgsl_composite_is_placeholder() {
        // Today the composite shader is embedded in stitch_renderer.
        // LShapeProjection returns "" until StitchCore migration
        // moves that shader source out through this trait.
        let p = LShapeProjection;
        assert!(p.wgsl_composite_source().is_empty());
    }

    // ---- CylindricalProjection (plan step 9) -------------------------------

    #[test]
    fn cylindrical_defaults_match_actionstitch() {
        // Reference values are the actionstitch-player defaults (focal
        // length 2400, 180-deg sweep, no screen tilt, normalized video
        // height). Regresses if someone silently changes the defaults.
        let c = CylindricalProjectionConfig::default();
        assert_eq!(c.focal_length, 2400.0);
        assert!((c.angular_sweep_rad - std::f32::consts::PI).abs() < 1e-6);
        assert_eq!(c.screen_rotation_rad, 0.0);
        assert_eq!(c.video_height, 1.0);
    }

    #[test]
    fn cylindrical_projection_reports_mono() {
        let p = CylindricalProjection::default();
        assert_eq!(p.name(), "cylindrical-mono-1camera");
        assert_eq!(
            p.camera_count(),
            1,
            "cylindrical projection consumes exactly one camera"
        );
    }

    #[test]
    fn cylindrical_theta_start_matches_actionstitch_formula() {
        // actionstitch's CylinderGeometry uses thetaStart = PI/2 - s/2
        // where s is the angular sweep. Verify for 180-deg (default)
        // and for a 90-deg cylinder (quarter sweep).
        let p180 = CylindricalProjection::default();
        assert!(
            (p180.theta_start_rad() - std::f32::consts::FRAC_PI_2 * 0.0).abs() < 1e-6,
            "180-deg sweep: theta_start = PI/2 - PI/2 = 0"
        );

        let p90 = CylindricalProjection::new(CylindricalProjectionConfig {
            angular_sweep_rad: std::f32::consts::FRAC_PI_2,
            ..Default::default()
        });
        // 90-deg: theta_start = PI/2 - PI/4 = PI/4.
        assert!((p90.theta_start_rad() - std::f32::consts::FRAC_PI_4).abs() < 1e-6);
    }

    #[test]
    fn cylindrical_wgsl_source_is_nonempty_and_has_expected_entrypoints() {
        // Sanity-check the embedded shader compiles in spirit: it must
        // declare both the vertex + fragment entry points the composite
        // pass expects. Full wgpu compilation lives in an integration
        // test behind the GPU gate.
        let p = CylindricalProjection::default();
        let src = p.wgsl_composite_source();
        assert!(!src.is_empty());
        assert!(src.contains("fn vs_fullscreen"));
        assert!(src.contains("fn fs_cylindrical_mono"));
        assert!(src.contains("CylUniforms"));
    }

    #[test]
    fn projection_dyn_dispatch_round_trip_with_mixed_camera_counts() {
        // Compile-time: `Box<dyn Projection>` can hold concrete impls
        // with different `camera_count()` results. Proves the trait
        // API's claim that consumers can swap projections without a
        // new type parameter on StitchCore.
        let projections: Vec<Box<dyn Projection>> = vec![
            Box::new(LShapeProjection),
            Box::new(CylindricalProjection::default()),
        ];
        assert_eq!(projections[0].camera_count(), 2);
        assert_eq!(projections[1].camera_count(), 1);
        assert_ne!(projections[0].name(), projections[1].name());
    }

    #[test]
    fn cylindrical_projection_is_send_sync() {
        fn assert_send_sync<T: Send + Sync + 'static>() {}
        assert_send_sync::<CylindricalProjection>();
        assert_send_sync::<CylindricalProjectionConfig>();
    }

    // `cylindrical_to_panorama` is the exact inverse of
    // `cylindrical_mono.wgsl`'s forward ray-cast, which I cannot execute
    // in a unit test (no GPU here). These check convention-agnostic
    // geometric invariants of the cylinder-at-origin construction
    // instead, so a sign error in either the shader or this function
    // would still very likely be caught: the vertical axis of the
    // cylinder is world Y, and the camera sits at the origin on that
    // axis, so any point at the vertical image center (v_norm = 0.5,
    // i.e. y_world = 0) must be exactly level (pitch = 0) regardless of
    // yaw convention, and left/right/top/bottom must be symmetric
    // around the center.

    #[test]
    fn cylindrical_to_panorama_vertical_center_is_level() {
        let config = CylindricalProjectionConfig::default();
        for norm_x in [0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            let pos = cylindrical_to_panorama(norm_x, 0.5, &config);
            assert!(
                pos.pitch.abs() < 1e-5,
                "vertical center (norm_x={norm_x}) must be level, got pitch={}",
                pos.pitch
            );
        }
    }

    #[test]
    fn cylindrical_to_panorama_left_right_symmetric_around_center() {
        let config = CylindricalProjectionConfig::default();
        let center = cylindrical_to_panorama(0.5, 0.5, &config);
        let left = cylindrical_to_panorama(0.0, 0.5, &config);
        let right = cylindrical_to_panorama(1.0, 0.5, &config);

        assert_ne!(
            left.yaw, right.yaw,
            "left/right edges must map to different yaw"
        );
        assert!(
            (left.yaw - center.yaw).abs() > 1e-3,
            "left edge must differ from center"
        );
        assert!(
            (right.yaw - center.yaw).abs() > 1e-3,
            "right edge must differ from center"
        );
        // 180-degree default sweep, centered: edges are symmetric around
        // the center yaw (opposite sign of displacement).
        assert!(
            ((left.yaw - center.yaw) + (right.yaw - center.yaw)).abs() < 1e-3,
            "left/right displacement from center must be symmetric: left={}, right={}, center={}",
            left.yaw,
            right.yaw,
            center.yaw
        );
    }

    #[test]
    fn cylindrical_to_panorama_top_bottom_symmetric_and_opposite_sign() {
        // Not `CylindricalProjectionConfig::default()`: its
        // `focal_length=2400` vs. `video_height=1.0` are world-unit
        // values meant to be set consistently by the actual consumer
        // (matching actionstitch's Three.js scene scale) - taken
        // literally together they subtend a near-zero vertical angle,
        // which would make this test's edges spuriously "almost
        // level" regardless of correctness. Any config with a
        // comparable focal_length/video_height ratio exercises the
        // same geometry meaningfully.
        let config = CylindricalProjectionConfig {
            focal_length: 1.0,
            video_height: 1.0,
            ..CylindricalProjectionConfig::default()
        };
        let top = cylindrical_to_panorama(0.5, 0.0, &config);
        let bottom = cylindrical_to_panorama(0.5, 1.0, &config);

        assert!(top.pitch.abs() > 1e-3, "top edge must not be level");
        assert!(
            (top.pitch + bottom.pitch).abs() < 1e-3,
            "top/bottom must be symmetric and opposite sign: top={}, bottom={}",
            top.pitch,
            bottom.pitch
        );
        assert_ne!(top.pitch.signum(), bottom.pitch.signum());
    }

    /// [`CylindricalProjectionConfig::video_height_for_aspect`] exists
    /// precisely to avoid the near-zero-vertical-angle trap the
    /// previous test's comment describes for the bare
    /// `focal_length=2400`/`video_height=1.0` default pairing - a 16:9
    /// source run through it must produce meaningful top/bottom pitch,
    /// not the ~0.01-degree sliver the unscaled default gives.
    #[test]
    fn video_height_for_aspect_gives_meaningful_pitch_range_for_default_focal_length() {
        let focal_length = CylindricalProjectionConfig::default().focal_length;
        let angular_sweep_rad = CylindricalProjectionConfig::default().angular_sweep_rad;
        let video_height = CylindricalProjectionConfig::video_height_for_aspect(
            focal_length,
            angular_sweep_rad,
            1920,
            1080,
        );
        let config = CylindricalProjectionConfig {
            video_height,
            ..CylindricalProjectionConfig::default()
        };

        let top = cylindrical_to_panorama(0.5, 0.0, &config);
        let bottom = cylindrical_to_panorama(0.5, 1.0, &config);

        // The unscaled default (video_height=1.0 against
        // focal_length=2400) gives top.pitch on the order of 1e-4 rad;
        // a correctly scaled 16:9 patch should give degrees, not
        // hundredths of a degree.
        assert!(
            top.pitch.abs() > 0.1,
            "expected a meaningful vertical FOV, got top.pitch={}",
            top.pitch
        );
        assert!((top.pitch + bottom.pitch).abs() < 1e-3);
    }

    #[test]
    fn cylindrical_panorama_bounds_are_symmetric_and_nonempty() {
        let focal_length = CylindricalProjectionConfig::default().focal_length;
        let angular_sweep_rad = CylindricalProjectionConfig::default().angular_sweep_rad;
        let video_height = CylindricalProjectionConfig::video_height_for_aspect(
            focal_length,
            angular_sweep_rad,
            1920,
            1080,
        );
        let config = CylindricalProjectionConfig {
            video_height,
            ..CylindricalProjectionConfig::default()
        };
        let ((yaw_min, yaw_max), (pitch_min, pitch_max)) = cylindrical_panorama_bounds(&config);

        assert!(yaw_min < 0.0 && yaw_max > 0.0, "yaw range must straddle center");
        assert!((yaw_min + yaw_max).abs() < 1e-3, "yaw bounds symmetric around 0");
        assert!(pitch_min < 0.0 && pitch_max > 0.0);
        assert!((pitch_min + pitch_max).abs() < 1e-3);
    }
}
