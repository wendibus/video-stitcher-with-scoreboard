//! End-to-end smoke test for [`reco_detect::CpuDetrDetector`] against a
//! real exported RF-DETR ONNX model.
//!
//! Ignored by default (no model ships in this repo - see `.gitignore`
//! and `scripts/export_rfdetr_onnx.py`). Run explicitly once a model is
//! exported locally:
//!
//! ```bash
//! /path/to/venv/bin/python3 scripts/export_rfdetr_onnx.py \
//!     --recomodel /path/to/Basketball-Small-*.recomodel \
//!     --out /tmp/basketball-rfdetr.onnx
//! RECO_DETR_TEST_MODEL=/tmp/basketball-rfdetr.onnx \
//!     cargo test -p reco-detect --test rfdetr_smoke -- --ignored --nocapture
//! ```
//!
//! This does not need real match footage: a synthetic mid-grey frame is
//! enough to exercise session load, preprocessing, inference, and
//! postprocessing end to end without panicking or erroring. It won't
//! assert on detection content (a grey frame has no ball to find) - the
//! unit tests in `detectors/mod.rs` already cover the decode math.

use reco_core::detect::detector::{
    CameraId, ChromaFormat, DetectorFrame, RawFrame, UnifiedDetector,
};
use reco_detect::CpuDetrDetector;

#[test]
#[ignore = "requires a real RF-DETR .onnx export; see module docs"]
fn detect_runs_end_to_end_against_real_model() {
    let model_path = std::env::var("RECO_DETR_TEST_MODEL")
        .expect("set RECO_DETR_TEST_MODEL to a real RF-DETR .onnx export (see module docs)");

    // The auto-dispatch `reco-autocam::setup_autocam` relies on must
    // actually classify this real export as RF-DETR, not stock YOLO.
    let (probe_session, _, _) =
        reco_detect::create_ort_session(std::path::Path::new(&model_path), Vec::new())
            .expect("failed to open model for shape probe");
    assert!(
        reco_detect::is_rf_detr_output_shape(&probe_session),
        "real RF-DETR export must be classified as RF-DETR by output shape"
    );
    drop(probe_session);

    let mut detector =
        CpuDetrDetector::with_config(&model_path, 0.1).expect("failed to load RF-DETR model");
    assert_eq!(detector.class_names(), &["ball".to_string()]);

    // Synthetic mid-grey 640x480 YUV420P frame - no ball to find, but
    // enough to prove the full pipeline runs without panicking/erroring.
    let (width, height) = (640u32, 480u32);
    let y = vec![128u8; (width * height) as usize];
    let u = vec![128u8; (width * height / 4) as usize];
    let v = vec![128u8; (width * height / 4) as usize];
    let frame = RawFrame {
        y: &y,
        chroma: ChromaFormat::Yuv420p { u: &u, v: &v },
        width,
        height,
    };

    let detections = detector
        .detect(CameraId::Left, &DetectorFrame::Cpu(frame))
        .expect("detection must not error against a real model + valid frame");
    println!("detections on synthetic grey frame: {detections:?}");
}
