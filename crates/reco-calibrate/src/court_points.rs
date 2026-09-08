//! Known FIBA basketball court reference points, in real-world meters.
//!
//! Used by [`crate::mono_optimizer`] to self-calibrate a single raw
//! camera's KB4 fisheye intrinsics + pose from a set of user-clicked
//! pixel correspondences against these known points - no checkerboard
//! shoot needed, since a full-court shot already contains enough known
//! geometry (corners, circles, arcs) to constrain the fit.
//!
//! ## Coordinate convention
//!
//! Origin at court center, floor level (`z=0` implicitly - points are
//! 2D since the court is flat). `+X` toward one baseline's right corner
//! (as seen from that end), `+Y` toward that same baseline. The court
//! is symmetric about both axes, so "which end is +Y" is arbitrary -
//! pick whichever end matches the LEFT side of the video frame when
//! clicking points in [`resources/court_calibration_editor.html`], and
//! be consistent.
//!
//! ## FIBA measurements used (2010+ rules)
//!
//! - Court: 28.0 x 15.0 m (half-length 14.0 m, half-width 7.5 m)
//! - Center circle radius: 1.80 m
//! - Free-throw line: 5.80 m from the baseline's inner edge
//! - Lane (key) width: 4.90 m (half-width 2.45 m)
//! - Three-point arc radius: 6.75 m
//! - Three-point arc center: 1.575 m from the baseline (the basket's
//!   floor projection), with a straight segment for the first 0.90 m
//!   of arc length from the baseline (the arc doesn't reach the
//!   sideline - approximated here with corner + arc-only samples,
//!   since the straight segment contributes little extra distortion
//!   signal over the corner points already present).

use std::f64::consts::PI;

/// One labeled point on the court floor plane, in meters.
///
/// See the module docs for the coordinate convention.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CourtPoint {
    pub label: &'static str,
    pub x_m: f64,
    pub y_m: f64,
}

const HALF_LENGTH: f64 = 14.0;
const HALF_WIDTH: f64 = 7.5;
const CENTER_CIRCLE_R: f64 = 1.8;
const FREE_THROW_DIST: f64 = 5.8;
const KEY_HALF_WIDTH: f64 = 2.45;
const THREE_POINT_R: f64 = 6.75;
const BASKET_OFFSET: f64 = 1.575;

/// Center-circle samples at 8 evenly-spaced angles - circles are
/// disproportionately useful for constraining radial distortion
/// (`k1`/`k2`): any residual barrel/pincushion warp bends a projected
/// circle into a non-ellipse in a way straight lines alone don't
/// expose as clearly.
fn center_circle_points() -> Vec<CourtPoint> {
    const LABELS: [&str; 8] = [
        "center-circle-0",
        "center-circle-45",
        "center-circle-90",
        "center-circle-135",
        "center-circle-180",
        "center-circle-225",
        "center-circle-270",
        "center-circle-315",
    ];
    LABELS
        .iter()
        .enumerate()
        .map(|(i, &label)| {
            let angle = i as f64 * PI / 4.0;
            CourtPoint {
                label,
                x_m: CENTER_CIRCLE_R * angle.cos(),
                y_m: CENTER_CIRCLE_R * angle.sin(),
            }
        })
        .collect()
}

/// Three-point arc samples on one end, at 3 angles spanning the arc
/// (excluding the near-sideline straight segment - see module docs).
///
/// `end_sign` is `+1.0` or `-1.0` selecting which baseline. Points are
/// parametrized by `phi`, the angle from the basket's "straight toward
/// mid-court" direction, swept left/right (`phi=0` is the arc's
/// midpoint, symmetric about the free-throw line's axis).
fn three_point_arc_points(end_sign: f64, end_label: &'static str) -> Vec<CourtPoint> {
    let basket_y = end_sign * (HALF_LENGTH - BASKET_OFFSET);
    let phis_deg = [-60.0_f64, 0.0, 60.0];
    let labels = ["-3pt-a", "-3pt-mid", "-3pt-b"];
    phis_deg
        .iter()
        .zip(labels)
        .map(|(&phi_deg, suffix)| {
            let phi = phi_deg.to_radians();
            CourtPoint {
                label: concat_label(end_label, suffix),
                x_m: THREE_POINT_R * phi.sin(),
                y_m: basket_y - end_sign * THREE_POINT_R * phi.cos(),
            }
        })
        .collect()
}

fn concat_label(end: &'static str, suffix: &str) -> &'static str {
    // Small fixed set of labels - leak a formatted string as &'static
    // str via Box::leak rather than threading lifetimes through the
    // whole module for what is always a handful of one-time-built
    // constant tables.
    Box::leak(format!("{end}{suffix}").into_boxed_str())
}

/// One baseline end's key/lane + free-throw-line corners plus the
/// basket floor-projection point.
fn end_points(end_sign: f64, end_label: &'static str) -> Vec<CourtPoint> {
    let baseline_y = end_sign * HALF_LENGTH;
    let free_throw_y = end_sign * (HALF_LENGTH - FREE_THROW_DIST);
    vec![
        CourtPoint {
            label: concat_label(end_label, "-corner-left"),
            x_m: -HALF_WIDTH,
            y_m: baseline_y,
        },
        CourtPoint {
            label: concat_label(end_label, "-corner-right"),
            x_m: HALF_WIDTH,
            y_m: baseline_y,
        },
        CourtPoint {
            label: concat_label(end_label, "-key-baseline-left"),
            x_m: -KEY_HALF_WIDTH,
            y_m: baseline_y,
        },
        CourtPoint {
            label: concat_label(end_label, "-key-baseline-right"),
            x_m: KEY_HALF_WIDTH,
            y_m: baseline_y,
        },
        CourtPoint {
            label: concat_label(end_label, "-key-freethrow-left"),
            x_m: -KEY_HALF_WIDTH,
            y_m: free_throw_y,
        },
        CourtPoint {
            label: concat_label(end_label, "-key-freethrow-right"),
            x_m: KEY_HALF_WIDTH,
            y_m: free_throw_y,
        },
        CourtPoint {
            label: concat_label(end_label, "-basket"),
            x_m: 0.0,
            y_m: end_sign * (HALF_LENGTH - BASKET_OFFSET),
        },
    ]
}

/// The full default point set: both baselines' corners/key/free-throw
/// points, both three-point arcs, the halfway line endpoints, and the
/// center circle.
///
/// Callers should trim this to whatever is actually visible in the
/// calibration frame before handing it to the point-picking tool (a
/// ceiling camera may not see the far end) - see the module docs.
pub fn default_points() -> Vec<CourtPoint> {
    let mut points = vec![
        CourtPoint {
            label: "halfway-left",
            x_m: -HALF_WIDTH,
            y_m: 0.0,
        },
        CourtPoint {
            label: "halfway-right",
            x_m: HALF_WIDTH,
            y_m: 0.0,
        },
    ];
    points.extend(center_circle_points());
    points.extend(end_points(1.0, "far"));
    points.extend(end_points(-1.0, "near"));
    points.extend(three_point_arc_points(1.0, "far"));
    points.extend(three_point_arc_points(-1.0, "near"));
    points
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_points_are_within_court_bounds() {
        for p in default_points() {
            assert!(
                p.x_m.abs() <= HALF_WIDTH + 1e-9,
                "{}: x={} exceeds half-width",
                p.label,
                p.x_m
            );
            assert!(
                p.y_m.abs() <= HALF_LENGTH + 1e-9,
                "{}: y={} exceeds half-length",
                p.label,
                p.y_m
            );
        }
    }

    #[test]
    fn default_points_has_no_duplicate_labels() {
        let points = default_points();
        let mut labels: Vec<&str> = points.iter().map(|p| p.label).collect();
        labels.sort_unstable();
        let mut deduped = labels.clone();
        deduped.dedup();
        assert_eq!(
            labels.len(),
            deduped.len(),
            "duplicate labels in default_points()"
        );
    }

    #[test]
    fn default_points_count_is_reasonable() {
        // 2 halfway + 8 center-circle + 7*2 end (corners/key/basket)
        // + 3*2 three-point arc = 30. More correspondence points only
        // strengthens the solve, as long as they're all visible in
        // frame - see the module docs about trimming to what's
        // actually visible before use.
        let count = default_points().len();
        assert!(
            (25..=35).contains(&count),
            "expected ~25-35 points, got {count}"
        );
    }

    #[test]
    fn center_circle_points_are_at_correct_radius() {
        for p in center_circle_points() {
            let r = (p.x_m * p.x_m + p.y_m * p.y_m).sqrt();
            assert!(
                (r - CENTER_CIRCLE_R).abs() < 1e-9,
                "{}: radius {r} != {CENTER_CIRCLE_R}",
                p.label
            );
        }
    }

    #[test]
    fn three_point_arc_points_are_at_correct_radius_from_basket() {
        let basket_far = (0.0, HALF_LENGTH - BASKET_OFFSET);
        for p in three_point_arc_points(1.0, "far") {
            let dx = p.x_m - basket_far.0;
            let dy = p.y_m - basket_far.1;
            let r = (dx * dx + dy * dy).sqrt();
            assert!(
                (r - THREE_POINT_R).abs() < 1e-6,
                "{}: radius from basket {r} != {THREE_POINT_R}",
                p.label
            );
        }
    }
}
