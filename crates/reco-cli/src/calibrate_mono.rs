//! `reco calibrate-mono` subcommand.
//!
//! Self-calibrates a single raw camera's KB4 fisheye intrinsics + pose
//! from user-clicked correspondences against known basketball court
//! geometry ([`reco_calibrate::court_points`]) - no checkerboard shoot
//! needed, since a full-court shot already contains enough known
//! structure (corners, circles, arcs) to constrain the fit. Output
//! feeds the mono pipeline's KB4 render path (Milestone 2, not yet
//! wired).
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
//! the court points itself. The frame is sent only to the given local
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
use reco_calibrate::mono_optimizer::{self, PointCorrespondence};
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

    let points = court_points::default_points();

    let clicked: Vec<ClickedPoint> = if let Some(path) = points_arg {
        // Points already supplied (e.g. re-running the solve on a
        // previously-saved points file) - skip both the browser and
        // any auto-labeling model entirely.
        let raw = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("reading {path}: {e}"))?;
        parse_clicked_points(&raw)?
    } else if let Some(model) = auto_model {
        println!("Asking {model} (via {ollama_endpoint}) to locate the court points - this can take a while for a large model...");
        let found = query_ollama_for_points(&png_bytes, &points, model, ollama_endpoint)?;
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
            base64::engine::general_purpose::STANDARD.encode(&png_bytes)
        );
        let labels_json =
            serde_json::to_string(&points.iter().map(|p| p.label).collect::<Vec<_>>())?;
        let html_path = write_editor_html(&data_uri, width, height, &labels_json)?;
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

    let result = mono_optimizer::optimize(&correspondences, width, height, max_iters)
        .map_err(|e| anyhow::anyhow!("calibration solve failed: {e}"))?;

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
             Check that clicked points match the printed labels, or re-run (multi-start \
             is randomness-free but sensitive to which points were clicked/skipped)."
        );
    }

    let json = serde_json::to_string_pretty(&result)?;
    std::fs::write(output, &json)?;
    println!("Wrote {output}");

    Ok(())
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

/// Prompt on stdin for the pasted points (path or raw JSON), retrying
/// on empty input or a bad file path instead of failing the whole run:
/// losing the browser-clicked work over one stray Enter would be a
/// bad trade for the cost of a loop here.
///
/// Only genuine end-of-input (stdin closed, e.g. piped from `/dev/null`
/// or a non-interactive script) gives up and returns an error.
fn read_points_from_stdin_with_retry() -> anyhow::Result<String> {
    loop {
        print!("> ");
        std::io::stdout().flush()?;
        let mut input = String::new();
        let bytes_read = std::io::stdin().read_line(&mut input)?;
        if bytes_read == 0 {
            anyhow::bail!("no input (stdin closed) - re-run and paste the points when prompted");
        }
        let input = input.trim();
        if input.is_empty() {
            println!("(empty - paste the file path or JSON, then press Enter)");
            continue;
        }
        if input.starts_with('{') {
            return Ok(input.to_string());
        }
        match std::fs::read_to_string(input) {
            Ok(contents) => return Ok(contents),
            Err(e) => {
                println!("Couldn't read '{input}': {e} - try again (path or raw JSON):");
                continue;
            }
        }
    }
}

/// Ask a local Ollama vision-language model to locate the labeled
/// court points in the calibration frame, replacing the manual
/// browser click-through.
///
/// Sends the frame to `endpoint` (a local Ollama server - nothing
/// leaves the machine) via `curl`, using the existing `ffmpeg`-via-
/// `Command` pattern (`calibration_io::extract_audio_pcm`) rather than
/// adding an HTTP client dependency for one call site.
fn query_ollama_for_points(
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

    let request_path = std::env::temp_dir().join(format!("reco_qwen_request_{}.json", std::process::id()));
    std::fs::write(&request_path, serde_json::to_vec(&request)?)
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

    let points: Vec<ClickedPoint> = serde_json::from_str(&parsed.response).map_err(|e| {
        anyhow::anyhow!("model did not return the expected JSON array ({e}): {}", parsed.response)
    })?;
    Ok(points)
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

fn write_editor_html(
    data_uri: &str,
    width: u32,
    height: u32,
    labels_json: &str,
) -> anyhow::Result<PathBuf> {
    // $HOME/.cache/reco instead of /tmp - snap-packaged browsers can't
    // access /tmp due to sandboxing (same rationale as reco-gui's ROI
    // editor, main.rs:1950-1951).
    let cache_base = std::env::var("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|_| std::env::var("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(|_| std::env::temp_dir());
    let dir = cache_base.join("reco").join("court_calibration");
    std::fs::create_dir_all(&dir)
        .map_err(|e| anyhow::anyhow!("creating {}: {e}", dir.display()))?;

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

/// Convert a YUV420P frame to PNG bytes (BT.709 limited range - same
/// matrix/range convention as `fisheye.wgsl`/`cylindrical_mono.wgsl`'s
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
