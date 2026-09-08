//! Single-camera KB4 fisheye self-calibration from known court points.
//!
//! Mirrors [`crate::optimizer`]'s multi-start Nelder-Mead pattern
//! (`CostFunction` impl, quadratic bounds penalty, multi-start loop
//! keeping the best result) - that pattern already converges
//! reliably on a 5-8 parameter stereo-layout problem across real
//! GoPro/DJI/XTU footage. This solves a different problem (one
//! camera's own fisheye intrinsics + pose from known planar court
//! points, instead of two cameras' relative layout from AKAZE-matched
//! features), but the optimization backend and defensive-programming
//! shape (bounds penalty, multi-start, keep-best) are deliberately
//! the same.
//!
//! # Why this is a harder convergence problem than the stereo case
//!
//! A single frame's ~20-30 point correspondences must simultaneously
//! pin down focal length, radial distortion, AND camera pose - unlike
//! the stereo case, there's no second viewpoint providing an
//! independent constraint. Monocular calibration from planar points
//! has a classic degeneracy: a larger focal length + a camera further
//! away can look nearly identical to a smaller focal length + closer
//! camera (the "bas-relief"-style ambiguity for a flat target). The
//! wide multi-start grid below exists specifically to fight this - it
//! is not a formality.

use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::neldermead::NelderMead;
use nalgebra::{Isometry3, Point3, Translation3, UnitQuaternion, Vector3};
use reco_core::calibration::CameraParams;
use reco_core::projection::project_world_point;

use crate::court_points::CourtPoint;
use crate::error::CalibrateError;

/// One clicked correspondence: a known court point and its normalized
/// `[0,1]` pixel position in the calibration frame.
#[derive(Debug, Clone, Copy)]
pub struct PointCorrespondence {
    pub court_point: CourtPoint,
    pub pixel: [f64; 2],
}

/// Result of a successful mono calibration solve.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MonoCalibrationResult {
    pub camera: CameraParams,
    /// Solved world -> camera rotation, axis-angle radians.
    pub rotation_axis_angle: [f64; 3],
    /// Solved camera position in court-plane world coordinates
    /// (meters); `z_m` is height above the floor. Not needed for
    /// rendering (only the rotation is), but reported for plausibility
    /// checks (e.g. "is this height sane for a ceiling mount?").
    pub camera_position_m: [f64; 3],
    pub mean_reprojection_error_px: f64,
}

/// Fixed distortion coefficients beyond `k1,k2` - see the module docs:
/// fitting more than 2 KB4 coefficients from a single frame risks
/// overfitting given the point count available.
const K3_K4: [f64; 2] = [0.0, 0.0];

const K_BOUND: (f64, f64) = (-0.5, 0.5);
const ROTATION_BOUND: (f64, f64) = (-std::f64::consts::PI, std::f64::consts::PI);
const LATERAL_BOUND: (f64, f64) = (-25.0, 25.0);
const HEIGHT_BOUND: (f64, f64) = (1.0, 40.0);

/// Penalty added per point that projects behind the camera / beyond
/// the (generous, unconstrained-during-solving) max field of view -
/// mirrors [`crate::optimizer`]'s `bounds_penalty` role of keeping
/// Nelder-Mead (which has no native constraint support) away from
/// degenerate parameter regions.
const MISSING_POINT_PENALTY: f64 = 10.0;

/// Field of view generous enough that a plausible fisheye lens's
/// actual max theta never gets accidentally clipped mid-solve (the
/// solved intrinsics' *real* usable field of view is a Milestone-2
/// concern, not something to constrain here).
const SOLVE_MAX_THETA_RAD: f64 = std::f64::consts::PI * 0.75;

fn f_seed_for_diagonal_fov_deg(diagonal_px: f64, fov_deg: f64) -> f64 {
    let theta_half = fov_deg.to_radians() * 0.5;
    (diagonal_px * 0.5) / theta_half
}

/// Build the `Isometry3` (world -> camera) for a given axis-angle
/// rotation and camera *position* in world coordinates.
///
/// Working in "camera position" terms (rather than the isometry's raw
/// translation, which is `-R * position`) keeps parameter bounds and
/// multi-start seeds physically meaningful (e.g. "height between 1
/// and 40 meters") instead of an opaque combination of rotation and
/// translation.
fn pose_from_rotation_and_position(
    rotation_axis_angle: Vector3<f64>,
    camera_position: Point3<f64>,
) -> Isometry3<f64> {
    let r = UnitQuaternion::from_scaled_axis(rotation_axis_angle);
    let t = Translation3::from(-(r * camera_position.coords));
    Isometry3::from_parts(t, r)
}

fn unpack(p: &[f64], width: u32, height: u32) -> (Isometry3<f64>, CameraParams) {
    let f = p[0];
    let camera = CameraParams {
        width,
        height,
        fx: f,
        fy: f,
        cx: width as f64 * 0.5,
        cy: height as f64 * 0.5,
        d: [p[1], p[2], K3_K4[0], K3_K4[1]],
    };
    let rotation = Vector3::new(p[3], p[4], p[5]);
    let position = Point3::new(p[6], p[7], p[8]);
    (pose_from_rotation_and_position(rotation, position), camera)
}

fn bounds_for(diagonal_px: f64) -> Vec<(f64, f64)> {
    // f bound: wide enough to cover the entire multi-start FOV grid
    // (see `f_seeds`) with margin on both ends.
    let f_lo = f_seed_for_diagonal_fov_deg(diagonal_px, 230.0) * 0.5;
    let f_hi = f_seed_for_diagonal_fov_deg(diagonal_px, 80.0) * 2.0;
    vec![
        (f_lo, f_hi),
        K_BOUND,
        K_BOUND,
        ROTATION_BOUND,
        ROTATION_BOUND,
        ROTATION_BOUND,
        LATERAL_BOUND,
        LATERAL_BOUND,
        HEIGHT_BOUND,
    ]
}

/// Quadratic penalty for parameters outside `bounds` - identical
/// shape to [`crate::optimizer`]'s `bounds_penalty`.
fn bounds_penalty(p: &[f64], bounds: &[(f64, f64)]) -> f64 {
    let scale = 1e4;
    let mut penalty = 0.0;
    for (&val, &(lo, hi)) in p.iter().zip(bounds) {
        if val < lo {
            let d = lo - val;
            penalty += scale * d * d;
        } else if val > hi {
            let d = val - hi;
            penalty += scale * d * d;
        }
    }
    penalty
}

struct MonoCalibrationCost<'a> {
    points: &'a [PointCorrespondence],
    width: u32,
    height: u32,
    bounds: Vec<(f64, f64)>,
}

impl Clone for MonoCalibrationCost<'_> {
    fn clone(&self) -> Self {
        Self {
            points: self.points,
            width: self.width,
            height: self.height,
            bounds: self.bounds.clone(),
        }
    }
}

impl CostFunction for MonoCalibrationCost<'_> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let (pose, camera) = unpack(p, self.width, self.height);
        let mut err = 0.0;
        for corr in self.points {
            let world_point = Point3::new(corr.court_point.x_m, corr.court_point.y_m, 0.0);
            match project_world_point(&pose, &world_point, &camera, SOLVE_MAX_THETA_RAD) {
                Some((px, py)) => {
                    err += (px - corr.pixel[0]).powi(2) + (py - corr.pixel[1]).powi(2)
                }
                None => err += MISSING_POINT_PENALTY,
            }
        }
        let mean_err = err / self.points.len().max(1) as f64;
        Ok(mean_err + bounds_penalty(p, &self.bounds))
    }
}

fn build_simplex(start: &[f64], bounds: &[(f64, f64)]) -> Vec<Vec<f64>> {
    const PERTURBATION: f64 = 0.10;
    let n = start.len();
    let mut vertices: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    vertices.push(start.to_vec());
    for i in 0..n {
        let mut vertex = start.to_vec();
        let range = bounds[i].1 - bounds[i].0;
        let delta = PERTURBATION * range;
        if vertex[i] + delta <= bounds[i].1 {
            vertex[i] += delta;
        } else {
            vertex[i] -= delta;
        }
        vertices.push(vertex);
    }
    vertices
}

fn run_nelder_mead(
    cost: &MonoCalibrationCost<'_>,
    start: &[f64],
    max_iters: u64,
) -> Option<(Vec<f64>, f64)> {
    let simplex = build_simplex(start, &cost.bounds);
    let solver: NelderMead<Vec<f64>, f64> =
        NelderMead::new(simplex).with_sd_tolerance(1e-12).ok()?;
    let res = Executor::new(cost.clone(), solver)
        .configure(|state| state.max_iters(max_iters))
        .run()
        .ok()?;
    let p = res.state().get_best_param()?.clone();
    let f = res.state().get_best_cost();
    Some((p, f))
}

/// Focal-length multi-start seeds, expressed as plausible diagonal
/// fields of view (degrees) - covers everything from a moderately
/// wide rectilinear-ish lens up to an extreme ceiling fisheye.
const FOV_SEEDS_DEG: [f64; 6] = [90.0, 120.0, 150.0, 175.0, 200.0, 220.0];

/// Camera-height seeds in meters (typical arena ceiling truss range).
const HEIGHT_SEEDS_M: [f64; 3] = [6.0, 9.0, 12.0];

/// Off-nadir tilt seeds in degrees (ceiling cams are rarely perfectly
/// straight down).
const TILT_SEEDS_DEG: [f64; 3] = [0.0, 15.0, 30.0];

/// Lateral position seeds: camera above court center, or offset
/// toward one baseline.
const LATERAL_SEEDS_M: [(f64, f64); 2] = [(0.0, 0.0), (0.0, 8.0)];

/// Rotation seeds near nadir (camera-local `+Z` -> world `-Z`), with
/// a small tilt swept in both directions around the world X axis.
/// Axis-angle `(pi, 0, 0)` is exact nadir (180 degree rotation about
/// X maps `+Z` to `-Z`); tilting by `tau` off nadir is approximated by
/// perturbing that angle by `+-tau` - Nelder-Mead refines the exact
/// rotation from there, so this only needs to land in the right basin,
/// not be geometrically exact.
fn rotation_seeds() -> Vec<Vector3<f64>> {
    let mut seeds = Vec::new();
    for &tilt_deg in &TILT_SEEDS_DEG {
        let tilt = tilt_deg.to_radians();
        if tilt_deg == 0.0 {
            seeds.push(Vector3::new(std::f64::consts::PI, 0.0, 0.0));
        } else {
            seeds.push(Vector3::new(std::f64::consts::PI - tilt, 0.0, 0.0));
            seeds.push(Vector3::new(std::f64::consts::PI + tilt, 0.0, 0.0));
        }
    }
    seeds
}

/// Solve for this camera's KB4 intrinsics + pose from a set of clicked
/// point correspondences against known court geometry.
///
/// `width`/`height` are the calibration frame's pixel dimensions
/// (used to fix the principal point at image center and to seed the
/// focal-length grid from plausible diagonal fields of view).
/// `max_iters` bounds each individual Nelder-Mead run (not the total
/// across the multi-start grid).
///
/// # Errors
///
/// [`CalibrateError::OptimizerFailed`] if every multi-start run fails
/// to produce a result (e.g. degenerate/too-few points).
pub fn optimize(
    points: &[PointCorrespondence],
    width: u32,
    height: u32,
    max_iters: u64,
) -> Result<MonoCalibrationResult, CalibrateError> {
    let diagonal_px = ((width * width + height * height) as f64).sqrt();
    let bounds = bounds_for(diagonal_px);
    let cost = MonoCalibrationCost {
        points,
        width,
        height,
        bounds: bounds.clone(),
    };

    let mut best: Option<(Vec<f64>, f64)> = None;
    for &fov_deg in &FOV_SEEDS_DEG {
        let f = f_seed_for_diagonal_fov_deg(diagonal_px, fov_deg);
        for rotation in rotation_seeds() {
            for &height_m in &HEIGHT_SEEDS_M {
                for &(lat_x, lat_y) in &LATERAL_SEEDS_M {
                    let start = vec![
                        f, 0.0, 0.0, rotation.x, rotation.y, rotation.z, lat_x, lat_y, height_m,
                    ];
                    if let Some((p, c)) = run_nelder_mead(&cost, &start, max_iters)
                        && best.as_ref().is_none_or(|(_, best_c)| c < *best_c)
                    {
                        best = Some((p, c));
                    }
                }
            }
        }
    }

    let (best_p, best_cost) = best.ok_or(CalibrateError::OptimizerFailed {
        max_evals: max_iters as usize,
    })?;
    let (pose, camera) = unpack(&best_p, width, height);
    let camera_world_position = pose.inverse().translation.vector;
    let _ = best_cost; // only used to pick the best multi-start run; reported error is recomputed in real pixels below
    let mean_reprojection_error_px = mean_pixel_error(points, &pose, &camera, width, height);

    Ok(MonoCalibrationResult {
        camera,
        rotation_axis_angle: [best_p[3], best_p[4], best_p[5]],
        camera_position_m: [
            camera_world_position.x,
            camera_world_position.y,
            camera_world_position.z,
        ],
        mean_reprojection_error_px,
    })
}

/// Mean Euclidean reprojection error in actual pixels (not the
/// optimizer's internal mixed-unit normalized cost) - the
/// human-facing number this module reports.
fn mean_pixel_error(
    points: &[PointCorrespondence],
    pose: &Isometry3<f64>,
    camera: &CameraParams,
    width: u32,
    height: u32,
) -> f64 {
    let mut sum = 0.0;
    let mut n = 0usize;
    for corr in points {
        let world_point = Point3::new(corr.court_point.x_m, corr.court_point.y_m, 0.0);
        if let Some((px, py)) =
            project_world_point(pose, &world_point, camera, SOLVE_MAX_THETA_RAD)
        {
            let dx = (px - corr.pixel[0]) * width as f64;
            let dy = (py - corr.pixel[1]) * height as f64;
            sum += (dx * dx + dy * dy).sqrt();
            n += 1;
        }
    }
    if n == 0 {
        f64::INFINITY
    } else {
        sum / n as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::court_points::default_points;

    /// Build synthetic, noise-free correspondences from a KNOWN ground
    /// -truth camera, then confirm the solver recovers it - the same
    /// verification style `optimizer.rs`'s `optimize_recovers_known_params`
    /// test uses.
    fn synthetic_correspondences(
        pose: &Isometry3<f64>,
        camera: &CameraParams,
    ) -> Vec<PointCorrespondence> {
        default_points()
            .into_iter()
            .filter_map(|cp| {
                let world_point = Point3::new(cp.x_m, cp.y_m, 0.0);
                project_world_point(pose, &world_point, camera, SOLVE_MAX_THETA_RAD).map(
                    |(px, py)| PointCorrespondence {
                        court_point: cp,
                        pixel: [px, py],
                    },
                )
            })
            .collect()
    }

    #[test]
    fn optimize_recovers_known_camera_from_synthetic_points() {
        let width = 3840u32;
        let height = 2160u32;
        let true_camera = CameraParams {
            width,
            height,
            fx: 1400.0,
            fy: 1400.0,
            cx: width as f64 * 0.5,
            cy: height as f64 * 0.5,
            d: [0.02, -0.01, 0.0, 0.0],
        };
        let true_rotation = Vector3::new(std::f64::consts::PI - 0.2, 0.0, 0.0);
        let true_position = Point3::new(0.0, 0.0, 9.0);
        let true_pose = pose_from_rotation_and_position(true_rotation, true_position);

        let points = synthetic_correspondences(&true_pose, &true_camera);
        assert!(
            points.len() >= 15,
            "expected most court points visible from a nadir-ish 9m mount, got {}",
            points.len()
        );

        let result = optimize(&points, width, height, 800).expect("optimization should succeed");

        assert!(
            result.mean_reprojection_error_px < 5.0,
            "reprojection error too high: {} px",
            result.mean_reprojection_error_px
        );
        assert!(
            (result.camera.fx - true_camera.fx).abs() < true_camera.fx * 0.1,
            "fx off: got {}, want {}",
            result.camera.fx,
            true_camera.fx
        );
        assert!(
            (result.camera_position_m[2] - 9.0).abs() < 2.0,
            "recovered height implausible: {}",
            result.camera_position_m[2]
        );
    }

    #[test]
    fn bounds_penalty_zero_inside() {
        let bounds = bounds_for(5000.0);
        let inside = vec![
            bounds[0].0 + 1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            10.0,
        ];
        assert_eq!(bounds_penalty(&inside, &bounds), 0.0);
    }

    #[test]
    fn bounds_penalty_nonzero_outside() {
        let bounds = bounds_for(5000.0);
        let outside = vec![bounds[0].0 - 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0];
        assert!(bounds_penalty(&outside, &bounds) > 0.0);
    }
}
