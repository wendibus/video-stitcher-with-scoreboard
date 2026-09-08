//! `reco calibrate-mono` subcommand.
//!
//! Self-calibrates a single raw camera's KB4 fisheye intrinsics + pose.
//! No checkerboard shoot needed - a full-court shot already contains
//! enough known structure to constrain the fit. Output feeds the mono
//! pipeline's KB4 render path.
//!
//! Default flow: click the court's 4 outer corners (any order) plus a
//! handful of points around the center circle (also any order) -
//! [`reco_calibrate::mono_optimizer::optimize_corners_and_circle`]
//! brute-forces which clicked corner is which (24 possible
//! assignments; a wrong one fits far worse than the right one, so this
//! is cheap and unambiguous) and fits the circle points as a shape
//! rather than named points. This replaced an earlier design that
//! asked for ~30 individually-labeled points (`far-corner-left`,
//! `near-3pt-a`, etc.) - real use showed keeping "which basket is far"
//! consistent across 30 clicks was the dominant source of bad
//! calibrations, not a shortage of correspondence points. `--legacy`
//! restores that labeled-point flow
//! ([`reco_calibrate::mono_optimizer::optimize`]) if the simplified
//! one ever proves insufficiently constrained for a given camera.
//!
//! Uses the same "extract frame -> template an HTML tool -> open in
//! browser -> read the result back" shape as `reco-gui`'s ROI editor
//! round-trip, but CLI-native: no Slint, and the point-picking tool
//! downloads a JSON file (`Blob` + `<a download>`) instead of using
//! the clipboard, since `arboard` is a `reco-gui`-only dependency and
//! a file download has no platform/headless edge cases.
//!
//! `--auto-model` replaces the manual click-through with a local
//! vision-language model (e.g. Qwen2.5-VL via Ollama) that locates
//! the points itself. The frame is sent only to the given local
//! Ollama endpoint (`http://localhost:11434` by default) via `curl` -
//! nothing leaves the machine. Point quality from a vision model is
//! unverified (this process never displays or otherwise inspects the
//! frame itself - see the workspace's standing privacy rule against
//! viewing local media), so treat it as a faster first draft to
//! spot-check, not a substitute for reviewing the reprojection error.

use std::io::Write as _;
use std::path::{Path, PathBuf};

use base64::Engine as _;
use reco_calibrate::court_points::{self, CourtPoint};
use reco_calibrate::mono_optimizer::{self, MonoCalibrationResult, PointCorrespondence};
use reco_io::ffmpeg::calibration_io;

/// Run the `calibrate-mono` subcommand.
#[allow(clippy::too_many_arguments)]
pub fn run_calibrate_mono(
    video: &str,
    frame_time_secs: f64,
    points_arg: Option<&str>,
    auto_model: Option<&str>,
    ollama_endpoint: &str,
    max_iters: u64,
    legacy: bool,
    output: &str,
) -> anyhow::Result<()> {
    reco_io::init();

    let probe = calibration_io::probe_video(Path::new(video))?;
    let frame_idx = (frame_time_secs * probe.fps).round().max(0.0) as u64;
    let frames = calibration_io::extract_frames(Path::new(video), &[frame_idx])?;
    let frame = frames
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("could not extract a frame at {frame_time_secs}s"))?;

    let (width, height) = (probe.width, probe.height);
    let png_bytes = yuv420p_to_png(&frame.y, &frame.u, &frame.v, width, height)?;

    let result = if legacy {
        run_legacy_flow(
            &png_bytes,
            points_arg,
            auto_model,
            ollama_endpoint,
            width,
            height,
            max_iters,
        )?
    } else {
        run_corners_circle_flow(
            &png_bytes,
            points_arg,
            auto_model,
            ollama_endpoint,
            width,
            height,
            max_iters,
        )?
    };

    println!(
        "Solved: fx=fy={:.1}px, k1={:.4}, k2={:.4}, mean reprojection error = {:.2}px",
        result.camera.fx, result.camera.d[0], result.camera.d[1], result.mean_reprojection_error_px
    );
    println!(
        "Recovered camera position (court-plane meters): x={:.2}, y={:.2}, height={:.2}",
        result.camera_position_m[0], result.camera_position_m[1], result.camera_position_m[2]
    );
    if result.mean_reprojection_error_px > 15.0 {
        println!(
            "WARNING: reprojection error is high - the solve may not have converged. \
             Check the clicked points, or re-run (multi-start is randomness-free but \
             sensitive to which points were clicked)."
        );
    }

    let json = serde_json::to_string_pretty(&result)?;
    std::fs::write(output, &json)?;
    println!("Wrote {output}");

    Ok(())
}

// ---------------------------------------------------------------------------
// Default flow: 4 unlabeled corners + unlabeled center-circle points.
// ---------------------------------------------------------------------------

#[derive(Debug, serde::Deserialize, serde::Serialize)]
struct CornersCircleFile {
    #[serde(default)]
    width: Option<u32>,
    #[serde(default)]
    height: Option<u32>,
    corners: Vec<[f64; 2]>,
    circle: Vec<[f64; 2]>,
}

fn run_corners_circle_flow(
    png_bytes: &[u8],
    points_arg: Option<&str>,
    auto_model: Option<&str>,
    ollama_endpoint: &str,
    width: u32,
    height: u32,
    max_iters: u64,
) -> anyhow::Result<MonoCalibrationResult> {
    let parsed: CornersCircleFile = if let Some(path) = points_arg {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("reading {path}: {e}"))?;
        serde_json::from_str(&raw).map_err(|e| anyhow::anyhow!("parsing {path}: {e}"))?
    } else if let Some(model) = auto_model {
        println!(
            "Asking {model} (via {ollama_endpoint}) to locate the court corners and center \
             circle - this can take a while for a large model..."
        );
        let found = query_ollama_for_corners_circle(png_bytes, model, ollama_endpoint)?;
        println!(
            "{model} found {} corners and {} circle points:",
            found.corners.len(),
            found.circle.len()
        );
        for c in &found.corners {
            println!("  corner  x={:.3} y={:.3}", c[0], c[1]);
        }
        for c in &found.circle {
            println!("  circle  x={:.3} y={:.3}", c[0], c[1]);
        }
        println!(
            "These were never verified against your actual footage (I don't view local \
             video/images - see the standing privacy rule) - spot-check a few against the \
             frame yourself before trusting the result."
        );
        found
    } else {
        let data_uri = format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(png_bytes)
        );
        let html_path = write_corners_editor_html(&data_uri, width, height)?;
        open::that(&html_path)
            .map_err(|e| anyhow::anyhow!("opening browser tool {}: {e}", html_path.display()))?;
        println!(
            "Browser opened ({}). Click the 4 court corners (any order), then a handful of \
             points around the center circle (any order, at least 5), then \"Save Points\" - \
             it downloads court_corners.json.\n\
             Paste either the full path to that file, or the raw JSON itself, below:",
            html_path.display()
        );
        let raw = read_points_from_stdin_with_retry()?;
        serde_json::from_str(&raw).map_err(|e| anyhow::anyhow!("parsing pasted JSON: {e}"))?
    };

    anyhow::ensure!(
        parsed.corners.len() == 4,
        "need exactly 4 court corners, got {}",
        parsed.corners.len()
    );
    anyhow::ensure!(
        parsed.circle.len() >= 5,
        "need at least 5 center-circle points, got {}",
        parsed.circle.len()
    );
    let corners: [[f64; 2]; 4] = parsed
        .corners
        .clone()
        .try_into()
        .map_err(|_| anyhow::anyhow!("expected exactly 4 corners"))?;

    println!(
        "Solving from 4 corners + {} circle points (trying all 24 possible corner orderings, \
         this can take up to a minute)...",
        parsed.circle.len()
    );
    mono_optimizer::optimize_corners_and_circle(&corners, &parsed.circle, width, height, max_iters)
        .map_err(|e| anyhow::anyhow!("calibration solve failed: {e}"))
}

/// Ask a local Ollama vision-language model to locate the 4 court
/// corners and a handful of center-circle points - a much simpler ask
/// than the legacy flow's ~30 individually-named points (no far/near/
/// left/right semantics to get right), so more likely to actually work
/// with a small local model.
fn query_ollama_for_corners_circle(
    png_bytes: &[u8],
    model: &str,
    endpoint: &str,
) -> anyhow::Result<CornersCircleFile> {
    let prompt = "You are analyzing a single frame from a ceiling-mounted camera above a \
         basketball court. The lens may have strong fisheye/barrel distortion, so court lines \
         can appear curved rather than straight.\n\n\
         Identify:\n\
         1. The pixel positions of the court's 4 outer corners (in any order).\n\
         2. About 8 points spread around the visible center circle of the court (in any \
         order).\n\n\
         Respond with ONLY JSON, no other text, in exactly this form:\n\
         {\"corners\": [[0.1,0.2],[0.9,0.2],[0.9,0.8],[0.1,0.8]], \"circle\": [[0.5,0.5], \
         [0.52,0.48], ...]}\n\
         where each pair is [x, y] as a fraction of image width/height (0.0 = left/top edge, \
         1.0 = right/bottom edge).";

    let request = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "images": [base64::engine::general_purpose::STANDARD.encode(png_bytes)],
        "format": "json",
        "stream": false,
    });

    let body = call_ollama_generate(&request, endpoint)?;
    let found: CornersCircleFile = serde_json::from_str(&body)
        .map_err(|e| anyhow::anyhow!("model did not return the expected JSON ({e}): {body}"))?;

    // Basic sanity filter, same rationale as the legacy flow's: a
    // point genuinely outside the image is a hallucination.
    let in_bounds = |p: &[f64; 2]| (-0.05..=1.05).contains(&p[0]) && (-0.05..=1.05).contains(&p[1]);
    let corners: Vec<[f64; 2]> = found.corners.into_iter().filter(in_bounds).collect();
    let circle: Vec<[f64; 2]> = found.circle.into_iter().filter(in_bounds).collect();
    Ok(CornersCircleFile {
        width: found.width,
        height: found.height,
        corners,
        circle,
    })
}

fn write_corners_editor_html(data_uri: &str, width: u32, height: u32) -> anyhow::Result<PathBuf> {
    let dir = calibration_cache_dir()?;
    let template = include_str!("../../../resources/court_corners_editor.html");
    let html = template
        .replace("{{IMAGE_DATA}}", data_uri)
        .replace("{{IMAGE_WIDTH}}", &width.to_string())
        .replace("{{IMAGE_HEIGHT}}", &height.to_string());
    let html_path = dir.join("court_corners_editor.html");
    std::fs::write(&html_path, html)
        .map_err(|e| anyhow::anyhow!("writing {}: {e}", html_path.display()))?;
    Ok(html_path)
}

// ---------------------------------------------------------------------------
// Legacy flow: ~30 individually-labeled points (far-corner-left, etc.)
// ---------------------------------------------------------------------------

fn run_legacy_flow(
    png_bytes: &[u8],
    points_arg: Option<&str>,
    auto_model: Option<&str>,
    ollama_endpoint: &str,
    width: u32,
    height: u32,
    max_iters: u64,
) -> anyhow::Result<MonoCalibrationResult> {
    let points = court_points::default_points();

    let clicked: Vec<ClickedPoint> = if let Some(path) = points_arg {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("reading {path}: {e}"))?;
        parse_clicked_points(&raw)?
    } else if let Some(model) = auto_model {
        println!("Asking {model} (via {ollama_endpoint}) to locate the court points - this can take a while for a large model...");
        let found = query_ollama_for_labeled_points(png_bytes, &points, model, ollama_endpoint)?;
        println!("{model} located {} of {} points:", found.len(), points.len());
        for p in &found {
            println!("  {:<28} x={:.3} y={:.3}", p.label, p.x, p.y);
        }
        println!(
            "These were never verified against your actual footage (I don't view local \
             video/images - see the standing privacy rule) - spot-check a few against the \
             frame yourself before trusting the result."
        );
        found
    } else {
        let data_uri = format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(png_bytes)
        );
        let labels_json =
            serde_json::to_string(&points.iter().map(|p| p.label).collect::<Vec<_>>())?;
        let html_path = write_labeled_editor_html(&data_uri, width, height, &labels_json)?;
        open::that(&html_path)
            .map_err(|e| anyhow::anyhow!("opening browser tool {}: {e}", html_path.display()))?;
        println!(
            "Browser opened ({}). Click through the labeled court points \
             (skip any not visible), then click \"Save Points\" - it downloads \
             court_points.json.\n\
             Paste either the full path to that file, or the raw JSON itself, below:",
            html_path.display()
        );
        let raw = read_points_from_stdin_with_retry()?;
        parse_clicked_points(&raw)?
    };

    anyhow::ensure!(
        clicked.len() >= 8,
        "only {} points available - need at least 8 for a stable solve (9 unknowns)",
        clicked.len()
    );

    let correspondences = match_clicked_to_known(&clicked, &points)?;
    println!(
        "Solving from {} point correspondences (this can take a few seconds)...",
        correspondences.len()
    );

    mono_optimizer::optimize(&correspondences, width, height, max_iters)
        .map_err(|e| anyhow::anyhow!("calibration solve failed: {e}"))
}

#[derive(Debug, serde::Deserialize)]
struct ClickedPointsFile {
    #[serde(default)]
    width: Option<u32>,
    #[serde(default)]
    height: Option<u32>,
    points: Vec<ClickedPoint>,
}

#[derive(Debug, serde::Deserialize)]
struct ClickedPoint {
    label: String,
    x: f64,
    y: f64,
}

/// Ask a local Ollama vision-language model to locate the labeled
/// court points in the calibration frame, replacing the manual
/// browser click-through.
///
/// Sends the frame to `endpoint` (a local Ollama server - nothing
/// leaves the machine) via `curl`, using the existing `ffmpeg`-via-
/// `Command` pattern (`calibration_io::extract_audio_pcm`) rather than
/// adding an HTTP client dependency for one call site.
fn query_ollama_for_labeled_points(
    png_bytes: &[u8],
    known: &[CourtPoint],
    model: &str,
    endpoint: &str,
) -> anyhow::Result<Vec<ClickedPoint>> {
    let labels: Vec<&str> = known.iter().map(|p| p.label).collect();
    let prompt = format!(
        "You are analyzing a single frame from a ceiling-mounted camera above a basketball \
         court. The lens may have strong fisheye/barrel distortion, so court lines can appear \
         curved rather than straight.\n\n\
         Identify the pixel location of as many of these labeled court reference points as are \
         visible. Naming convention:\n\
         - \"halfway-left\"/\"halfway-right\": where the center line meets the two sidelines.\n\
         - \"center-circle-N\" (N in degrees 0..315): points around the center circle.\n\
         - \"far-*\"/\"near-*\": the same set of points at each basket end - you choose which \
         end is \"far\" vs \"near\", but be consistent across every far-*/near-* label:\n\
         \x20\x20- \"-corner-left\"/\"-corner-right\": that end's two baseline corners.\n\
         \x20\x20- \"-key-baseline-left\"/\"-key-baseline-right\": where the free-throw lane \
         meets the baseline.\n\
         \x20\x20- \"-key-freethrow-left\"/\"-key-freethrow-right\": the lane's two corners at \
         the free-throw line.\n\
         \x20\x20- \"-basket\": the point on the floor directly under the basket.\n\
         \x20\x20- \"-3pt-a\"/\"-3pt-mid\"/\"-3pt-b\": three points along the three-point arc \
         (mid = straight out from the basket toward mid-court; a/b = left/right of that).\n\
         For \"left\"/\"right\", use whichever side is on the left/right of THIS image, but stay \
         consistent across every point.\n\n\
         Labels to locate: {}\n\n\
         Respond with ONLY a JSON array, no other text, in exactly this form:\n\
         [{{\"label\": \"halfway-left\", \"x\": 0.12, \"y\": 0.55}}, ...]\n\
         where x and y are the point's position as a fraction of image width/height (0.0 = \
         left/top edge, 1.0 = right/bottom edge). Omit any label not visible in the frame. Do \
         not invent labels outside the given list.",
        labels.join(", ")
    );

    let request = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "images": [base64::engine::general_purpose::STANDARD.encode(png_bytes)],
        "format": "json",
        "stream": false,
    });

    let body = call_ollama_generate(&request, endpoint)?;
    let points = parse_model_points(&body)?;

    // Basic sanity filter: a point genuinely outside the image (small
    // tolerance for edge rounding) is a hallucination, not a
    // borderline-visible point - drop it rather than feed the solver
    // garbage silently.
    let (valid, invalid): (Vec<_>, Vec<_>) = points
        .into_iter()
        .partition(|p| (-0.05..=1.05).contains(&p.x) && (-0.05..=1.05).contains(&p.y));
    for p in &invalid {
        log::warn!(
            "dropping '{}' from {model}: coordinates out of image bounds (x={}, y={})",
            p.label,
            p.x,
            p.y
        );
    }
    Ok(valid)
}

/// POST `request` to `endpoint`'s `/api/generate` via `curl` and
/// return the model's raw text response (the `"response"` field of
/// Ollama's JSON envelope).
fn call_ollama_generate(request: &serde_json::Value, endpoint: &str) -> anyhow::Result<String> {
    let request_path =
        std::env::temp_dir().join(format!("reco_ollama_request_{}.json", std::process::id()));
    std::fs::write(&request_path, serde_json::to_vec(request)?)
        .map_err(|e| anyhow::anyhow!("writing Ollama request: {e}"))?;

    let url = format!("{}/api/generate", endpoint.trim_end_matches('/'));
    let output = std::process::Command::new("curl")
        .args([
            "-s",
            "--max-time",
            "600",
            "-X",
            "POST",
            &url,
            "-H",
            "Content-Type: application/json",
            "-d",
        ])
        .arg(format!("@{}", request_path.display()))
        .output();
    let _ = std::fs::remove_file(&request_path);
    let output = output.map_err(|e| anyhow::anyhow!("running curl against {url}: {e}"))?;
    anyhow::ensure!(
        output.status.success(),
        "curl against {url} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let body = String::from_utf8_lossy(&output.stdout);
    #[derive(serde::Deserialize)]
    struct OllamaGenerateResponse {
        response: String,
    }
    let parsed: OllamaGenerateResponse = serde_json::from_str(&body)
        .map_err(|e| anyhow::anyhow!("parsing Ollama response ({e}): {body}"))?;
    Ok(parsed.response)
}

/// Parse a model's point response in either shape it might use:
/// `[{"label":..., "x":..., "y":...}, ...]` or `{"label": [x, y], ...}`
/// (some models default to the latter despite the prompted schema).
fn parse_model_points(response: &str) -> anyhow::Result<Vec<ClickedPoint>> {
    if let Ok(points) = serde_json::from_str::<Vec<ClickedPoint>>(response) {
        return Ok(points);
    }
    let as_map: std::collections::HashMap<String, [f64; 2]> = serde_json::from_str(response)
        .map_err(|e| {
            anyhow::anyhow!(
                "model did not return the expected JSON array or object ({e}): {response}"
            )
        })?;
    Ok(as_map
        .into_iter()
        .map(|(label, [x, y])| ClickedPoint { label, x, y })
        .collect())
}

fn parse_clicked_points(raw: &str) -> anyhow::Result<Vec<ClickedPoint>> {
    let file: ClickedPointsFile =
        serde_json::from_str(raw).map_err(|e| anyhow::anyhow!("parsing clicked points: {e}"))?;
    let _ = (file.width, file.height); // informational only; solve uses the probed frame dims
    Ok(file.points)
}

/// Match clicked `{label, x, y}` entries against the known court
/// point catalog by label, dropping (with a warning) any clicked
/// label that isn't recognized.
fn match_clicked_to_known(
    clicked: &[ClickedPoint],
    known: &[CourtPoint],
) -> anyhow::Result<Vec<PointCorrespondence>> {
    let mut out = Vec::with_capacity(clicked.len());
    for c in clicked {
        match known.iter().find(|k| k.label == c.label) {
            Some(&court_point) => out.push(PointCorrespondence {
                court_point,
                pixel: [c.x, c.y],
            }),
            None => log::warn!("unrecognized point label '{}' - ignoring", c.label),
        }
    }
    anyhow::ensure!(!out.is_empty(), "no clicked points matched known labels");
    Ok(out)
}

fn write_labeled_editor_html(
    data_uri: &str,
    width: u32,
    height: u32,
    labels_json: &str,
) -> anyhow::Result<PathBuf> {
    let dir = calibration_cache_dir()?;
    let template = include_str!("../../../resources/court_calibration_editor.html");
    let html = template
        .replace("{{IMAGE_DATA}}", data_uri)
        .replace("{{IMAGE_WIDTH}}", &width.to_string())
        .replace("{{IMAGE_HEIGHT}}", &height.to_string())
        .replace("{{POINT_LABELS_JSON}}", labels_json);

    let html_path = dir.join("court_calibration_editor.html");
    std::fs::write(&html_path, html)
        .map_err(|e| anyhow::anyhow!("writing {}: {e}", html_path.display()))?;
    Ok(html_path)
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Prompt on stdin for the pasted points (path or raw JSON), retrying
/// on empty input or a bad file path instead of failing the whole run:
/// losing the browser-clicked work over one stray Enter would be a
/// bad trade for the cost of a loop here.
///
/// The pretty-printed JSON this tool's own `saveAndClose()` produces
/// spans many lines - a naive single `read_line()` call only captures
/// the first (`"{"`) and, worse, exits *before* the rest of the pasted
/// text has been consumed, so the terminal's shell ends up "typing"
/// the remaining JSON lines as commands once this process exits. Once
/// a line starts an object/array, this keeps reading and
/// re-attempting to parse the accumulated buffer as JSON until it
/// succeeds, rather than assuming one line is the whole document.
///
/// Only genuine end-of-input (stdin closed, e.g. piped from `/dev/null`
/// or a non-interactive script) gives up and returns an error.
fn read_points_from_stdin_with_retry() -> anyhow::Result<String> {
    let stdin = std::io::stdin();
    read_points_from_reader_with_retry(&mut stdin.lock())
}

/// Reader-injected core of [`read_points_from_stdin_with_retry`] - takes
/// any [`std::io::BufRead`] so this is unit-testable against an
/// in-memory buffer instead of real stdin.
fn read_points_from_reader_with_retry(reader: &mut impl std::io::BufRead) -> anyhow::Result<String> {
    loop {
        print!("> ");
        std::io::stdout().flush()?;
        let mut input = String::new();
        let bytes_read = reader.read_line(&mut input)?;
        if bytes_read == 0 {
            anyhow::bail!("no input (stdin closed) - re-run and paste the points when prompted");
        }
        let trimmed = input.trim();
        if trimmed.is_empty() {
            println!("(empty - paste the file path or JSON, then press Enter)");
            continue;
        }
        if trimmed.starts_with('{') || trimmed.starts_with('[') {
            match read_multiline_json(reader, input) {
                Ok(json) => return Ok(json),
                Err(e) => {
                    println!("{e} - try again (path or raw JSON):");
                    continue;
                }
            }
        }
        match std::fs::read_to_string(trimmed) {
            Ok(contents) => return Ok(contents),
            Err(e) => {
                println!("Couldn't read '{trimmed}': {e} - try again (path or raw JSON):");
                continue;
            }
        }
    }
}

/// Keep reading lines from `reader` onto `buffer` (which already holds
/// the first line) until it parses as valid JSON, up to a generous
/// line cap. Pasted terminal input arrives already queued on stdin, so
/// these `read_line` calls return immediately rather than blocking on
/// the user typing further - this is not an interactive multi-line
/// editor.
///
/// Exists because the pretty-printed JSON this tool's own
/// `saveAndClose()` produces spans many lines - a naive single
/// `read_line()` call only captures the first (`"{"`) and, worse,
/// returns to the caller *before* the rest of the pasted text has been
/// consumed, so the terminal's shell ends up "typing" the remaining
/// JSON lines as commands once this process exits/prompts again.
fn read_multiline_json(reader: &mut impl std::io::BufRead, mut buffer: String) -> anyhow::Result<String> {
    const MAX_EXTRA_LINES: usize = 2000;
    if serde_json::from_str::<serde_json::Value>(&buffer).is_ok() {
        return Ok(buffer);
    }
    for _ in 0..MAX_EXTRA_LINES {
        let mut line = String::new();
        let bytes_read = reader.read_line(&mut line)?;
        if bytes_read == 0 {
            break; // stdin closed mid-paste
        }
        buffer.push_str(&line);
        if serde_json::from_str::<serde_json::Value>(&buffer).is_ok() {
            return Ok(buffer);
        }
    }
    Err(anyhow::anyhow!(
        "pasted text never became valid JSON (read up to {MAX_EXTRA_LINES} extra lines)"
    ))
}

#[cfg(test)]
mod stdin_tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn reads_pretty_printed_multiline_json_pasted_at_once() {
        let pasted = "{\n  \"width\": 1920,\n  \"height\": 1080,\n  \"corners\": [[0.1,0.1]],\n  \"circle\": [[0.5,0.5]]\n}\n";
        let mut reader = Cursor::new(pasted.as_bytes());
        let json = read_points_from_reader_with_retry(&mut reader).expect("should parse");
        let parsed: CornersCircleFile = serde_json::from_str(&json).expect("should be valid");
        assert_eq!(parsed.width, Some(1920));
        assert_eq!(parsed.corners.len(), 1);
    }

    #[test]
    fn reads_single_line_json_unchanged() {
        let pasted = "{\"width\":1920,\"height\":1080,\"corners\":[],\"circle\":[]}\n";
        let mut reader = Cursor::new(pasted.as_bytes());
        let json = read_points_from_reader_with_retry(&mut reader).expect("should parse");
        assert!(serde_json::from_str::<CornersCircleFile>(&json).is_ok());
    }

    #[test]
    fn retries_past_a_blank_line_before_the_real_input() {
        let pasted = "\n/does/not/exist/at/all.json\n{\"a\":1}\n";
        let mut reader = Cursor::new(pasted.as_bytes());
        let result = read_points_from_reader_with_retry(&mut reader).expect("should eventually succeed");
        assert_eq!(result.trim(), "{\"a\":1}");
    }

    #[test]
    fn eof_with_no_input_is_an_error_not_a_hang() {
        let mut reader = Cursor::new(&b""[..]);
        assert!(read_points_from_reader_with_retry(&mut reader).is_err());
    }
}

/// $HOME/.cache/reco/court_calibration - snap-packaged browsers can't
/// access /tmp due to sandboxing (same rationale as reco-gui's ROI
/// editor).
fn calibration_cache_dir() -> anyhow::Result<PathBuf> {
    let cache_base = std::env::var("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|_| std::env::var("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(|_| std::env::temp_dir());
    let dir = cache_base.join("reco").join("court_calibration");
    std::fs::create_dir_all(&dir).map_err(|e| anyhow::anyhow!("creating {}: {e}", dir.display()))?;
    Ok(dir)
}

/// Convert a YUV420P frame to PNG bytes (BT.709 limited range - same
/// matrix/range convention as `fisheye.wgsl`/`kb4_mono.wgsl`'s
/// `sample_yuv`, so the calibration frame looks the same as the
/// rendered output).
fn yuv420p_to_png(y: &[u8], u: &[u8], v: &[u8], width: u32, height: u32) -> anyhow::Result<Vec<u8>> {
    let (w, h) = (width as usize, height as usize);
    let mut rgb = vec![0u8; w * h * 3];
    for row in 0..h {
        for col in 0..w {
            let y_raw = y[row * w + col] as f64;
            let cu = col / 2;
            let cr = row / 2;
            let chroma_w = w.div_ceil(2);
            let u_raw = u[cr * chroma_w + cu] as f64;
            let v_raw = v[cr * chroma_w + cu] as f64;

            let yy = (y_raw - 16.0) * (255.0 / 219.0);
            let cb = (u_raw - 128.0) * (255.0 / 224.0);
            let cc = (v_raw - 128.0) * (255.0 / 224.0);

            let r = yy + 1.5748 * cc;
            let g = yy - 0.1873 * cb - 0.4681 * cc;
            let b = yy + 1.8556 * cb;

            let idx = (row * w + col) * 3;
            rgb[idx] = r.clamp(0.0, 255.0) as u8;
            rgb[idx + 1] = g.clamp(0.0, 255.0) as u8;
            rgb[idx + 2] = b.clamp(0.0, 255.0) as u8;
        }
    }

    let mut png_bytes = Vec::new();
    {
        let mut cursor = std::io::Cursor::new(&mut png_bytes);
        image::write_buffer_with_format(
            &mut cursor,
            &rgb,
            width,
            height,
            image::ExtendedColorType::Rgb8,
            image::ImageFormat::Png,
        )?;
    }
    Ok(png_bytes)
}
