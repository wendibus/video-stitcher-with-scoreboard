//! Mono subcommand: track a ball in a single-camera (not yet
//! stitched/projected) video and encode the followed crop.
//!
//! Uses `MonoJob` (Layer 3 API, `reco-io`). No calibration file: the
//! cylindrical projection needs only lens/geometry parameters, not a
//! two-camera stereo calibration - see `MonoStitchCoreConfig`.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

/// Arguments for the mono subcommand, collected from CLI parsing.
pub struct MonoArgs<'a> {
    pub input: &'a str,
    pub output: &'a str,
    pub width: u32,
    pub height: u32,
    pub encoder_name: Option<String>,
    pub codec: &'a str,
    pub quality: &'a str,
    pub quality_value: Option<u8>,
    pub preset: Option<String>,
    pub no_audio: bool,
    pub model_path: Option<&'a str>,
    pub detection_interval: u64,
    pub tracking_mode: &'a str,
    pub panner_preset: Option<&'a str>,
    pub panner_config_path: Option<&'a str>,
    pub allow_no_tracking: bool,
}

/// Run the mono subcommand.
pub fn run_mono(args: MonoArgs<'_>, interrupted: &Arc<AtomicBool>) -> anyhow::Result<()> {
    const MAX_DIM: u32 = 8192;
    anyhow::ensure!(
        args.width > 0 && args.width <= MAX_DIM && args.height > 0 && args.height <= MAX_DIM,
        "Output dimensions {}x{} out of range: width and height must be 1..={MAX_DIM}",
        args.width,
        args.height,
    );

    if args.tracking_mode != "sweep" {
        anyhow::ensure!(
            args.model_path.is_some(),
            "mono requires --model unless --tracking sweep (there is no default detector)"
        );
    }
    if let Some(model_path) = args.model_path {
        reco_autocam::validate_model_path(std::path::Path::new(model_path))?;
    }

    let codec: reco_io::output::Codec = args
        .codec
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid --codec: {e}"))?;
    let quality: reco_io::output::Quality = args
        .quality
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid --quality: {e}"))?;

    let mut job = reco_io::MonoJob::new(args.input, args.output)
        .codec(codec)
        .quality(quality)
        .resolution(args.width, args.height)
        .detection_interval(args.detection_interval)
        .with_audio(!args.no_audio);
    if let Some(name) = args.encoder_name {
        job = job.encoder_name(name);
    }
    if let Some(q) = args.quality_value {
        job = job.quality_value(q);
    }
    if let Some(p) = args.preset {
        job = job.preset(p);
    }

    // Resolve FieldPanner tuning up front so a bad preset/file fails
    // before decoding starts (same rationale as `stitch::run_stitch`).
    let panner_cfg: Option<reco_autocam::panners::FieldPannerConfig> = {
        use reco_autocam::panners::{FieldPannerConfig, PRESET_NAMES};
        let base = match args.panner_preset {
            Some(name) => {
                let c = FieldPannerConfig::from_preset_name(name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "unknown --panner-preset '{name}' (expected: {})",
                        PRESET_NAMES.join(", ")
                    )
                })?;
                log::info!("FieldPanner preset: {name}");
                Some(c)
            }
            None => None,
        };
        match args.panner_config_path {
            Some(p) => {
                let contents = std::fs::read_to_string(p)
                    .map_err(|e| anyhow::anyhow!("reading panner config {p}: {e}"))?;
                let cfg = if let Some(b) = base {
                    let mut v = serde_json::to_value(&b).expect("config serializes");
                    let over: serde_json::Value = serde_json::from_str(&contents)
                        .map_err(|e| anyhow::anyhow!("parsing panner config {p}: {e}"))?;
                    if let (Some(bm), serde_json::Value::Object(om)) = (v.as_object_mut(), over) {
                        bm.extend(om);
                    }
                    serde_json::from_value(v)
                        .map_err(|e| anyhow::anyhow!("applying panner config {p}: {e}"))?
                } else {
                    serde_json::from_str(&contents)
                        .map_err(|e| anyhow::anyhow!("parsing panner config {p}: {e}"))?
                };
                log::info!("FieldPanner config loaded from {p}");
                Some(cfg)
            }
            None => base,
        }
    };

    let tracking_failed = Arc::new(std::sync::atomic::AtomicBool::new(false));

    if args.tracking_mode == "sweep" {
        job = job.on_session(|core, _fps| {
            let panner = Box::new(reco_autocam::panners::SweepPanner::new(0.8, 10.0));
            core.set_panner(panner);
            log::info!("Tracking mode: sweep (debug, no AI)");
        });
    } else {
        let model_path = args.model_path.expect("checked above").to_owned();
        let interval = args.detection_interval;
        let mode_str = args.tracking_mode.to_owned();
        let allow_fallback = args.allow_no_tracking;
        let tracking_failed_hook = Arc::clone(&tracking_failed);
        job = job.on_session(move |core, fps| {
            let mode = match mode_str.as_str() {
                "ball" => reco_autocam::TrackingMode::Ball,
                _ => reco_autocam::TrackingMode::Field,
            };
            let mut autocam_config = reco_autocam::AutocamConfig::new(&model_path)
                .with_tracking_mode(mode)
                .with_detection_interval(interval);
            if mode == reco_autocam::TrackingMode::Ball {
                autocam_config.confidence_threshold = Some(0.25);
            }
            if let Some(ref cfg) = panner_cfg {
                autocam_config.field_panner_config = Some(cfg.clone());
            }
            // Mono only supports CPU decode today (`source_is_gpu_resident=false`)
            // - see MonoJob's module docs.
            match reco_autocam::setup_autocam(core, &autocam_config, fps as f32, false) {
                Ok(true) => println!("Autocam: tracking enabled (model: {model_path})"),
                Ok(false) => {
                    let msg = "Tracking requested but no CPU detector backend is compiled in. \
                               Pass --allow-no-tracking to continue without tracking.";
                    if allow_fallback {
                        log::warn!("{msg}");
                    } else {
                        log::error!("{msg}");
                        tracking_failed_hook.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                }
                Err(e) => {
                    let msg = format!(
                        "Autocam setup failed: {e}. Pass --allow-no-tracking to continue without tracking."
                    );
                    if allow_fallback {
                        log::warn!("{msg}");
                    } else {
                        log::error!("{msg}");
                        tracking_failed_hook.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            }
        });
    }

    let result = job.run(interrupted)?;
    anyhow::ensure!(
        !tracking_failed.load(std::sync::atomic::Ordering::Relaxed),
        "tracking setup failed (see above); pass --allow-no-tracking to continue without it"
    );
    println!("Mono: {} frames -> {}", result.frame_count, args.output);
    Ok(())
}
