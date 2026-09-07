//! RF-DETR ONNX detector for sport-specific ball detection on raw camera frames.
//!
//! Runs an RF-DETR model exported via `scripts/export_rfdetr_onnx.py`
//! (no letterbox, no NMS, two named outputs `dets`/`labels`). Unlike
//! [`super::cpu::CpuYoloDetector`], RF-DETR is trained on a plain
//! (aspect-distorting) resize, so preprocessing skips the grey-pad
//! letterbox entirely, and postprocessing needs no un-letterbox step -
//! normalized box coordinates from the model already are normalized
//! frame coordinates.
//!
//! ## Canonical preprocessing spec (RF-DETR only - differs from YOLO's)
//!
//! - **Color**: BT.601 full-range YUV -> RGB (same convention as every
//!   other reco-detect backend, see [`super::bt601_yuv_to_rgb`])
//! - **Resize**: bilinear interpolation, **no letterbox** - stretched
//!   directly to `resolution x resolution`
//! - **Normalize**: ImageNet mean/std (`[0.485,0.456,0.406]` /
//!   `[0.229,0.224,0.225]`), matching `rfdetr`'s own training/export
//!   pipeline
//! - **Layout**: CHW float32, `[1, 3, H, W]`

use std::path::Path;

use ort::session::Session;
use ort::value::TensorRef;
use reco_core::detect::detector::{
    CameraId, Detection, DetectorError, DetectorFrame, RawFrame, UnifiedDetector,
};

use super::{bt601_yuv_to_rgb, chroma_sample, postprocess_detr};

/// ImageNet normalization constants RF-DETR was trained/exported with.
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// RF-DETR-based object detector using ONNX Runtime on CPU.
///
/// Loads an RF-DETR ONNX export (`dets`: `[1, Q, 4]` cxcywh boxes,
/// `labels`: `[1, Q, C]` raw per-class logits) and runs inference on
/// raw camera frames. Only single-class ball models are supported
/// (class 0 = ball, remaining class slot(s) = RF-DETR's background
/// slot) - see [`postprocess_detr`].
pub struct CpuDetrDetector {
    session: Session,
    input_size: u32,
    confidence_threshold: f32,
    /// Always `["ball"]` - RF-DETR exports carry no Ultralytics-style
    /// `names` metadata, and this detector only supports single-class
    /// ball models (see [`postprocess_detr`]). Kept as an owned `Vec`
    /// so [`UnifiedDetector::class_names`] can hand out a `&[String]`
    /// consistent with every other backend.
    labels: Vec<String>,
    /// Pre-allocated preprocess scratch for `3 * input_size * input_size` f32.
    rgb_chw_buf: Vec<f32>,
}

impl CpuDetrDetector {
    /// Load an RF-DETR ONNX model with a confidence threshold.
    ///
    /// `input_size` is read from the model's own BCHW input shape (see
    /// [`crate::ort_session::create_ort_session`]); labels are not
    /// read from ONNX metadata (RF-DETR exports carry none) - the
    /// single tracked class is always `"ball"`.
    pub fn with_config(
        path: impl AsRef<Path>,
        confidence_threshold: f32,
    ) -> Result<Self, crate::ort_session::SessionError> {
        let (session, input_size, labels) =
            crate::ort_session::create_ort_session(path.as_ref(), vec!["ball".to_string()])?;

        log::info!(
            "CpuDetrDetector loaded: input={}x{}, conf_thresh={}",
            input_size,
            input_size,
            confidence_threshold,
        );

        let sz = input_size as usize;
        Ok(Self {
            session,
            input_size,
            confidence_threshold,
            labels,
            rgb_chw_buf: vec![0.0_f32; 3 * sz * sz],
        })
    }

    /// Class names from the model (always `["ball"]` - see [`Self::labels`]).
    pub fn class_names(&self) -> &[String] {
        &self.labels
    }

    /// Model input size (square dimension, e.g. 512).
    pub fn input_size(&self) -> u32 {
        self.input_size
    }

    /// Fill `self.rgb_chw_buf` from a raw YUV frame: flat RGB float32 in
    /// CHW layout, stretched (no letterbox) to `input_size x
    /// input_size`, ImageNet-normalized. Reused across frames - no
    /// allocation happens per call.
    fn preprocess(&mut self, frame: &RawFrame<'_>) {
        let sz = self.input_size as usize;
        let plane = sz * sz;
        let w_max = frame.width - 1;
        let h_max = frame.height - 1;

        // Plain resize: destination pixel -> fractional source
        // coordinate, scaled independently per axis (RF-DETR was
        // exported the same way, so this is not a distortion bug -
        // see the module docs).
        let x_scale = frame.width as f32 / sz as f32;
        let y_scale = frame.height as f32 / sz as f32;

        for dy in 0..sz as u32 {
            for dx in 0..sz as u32 {
                let src_x = (dx as f32 + 0.5) * x_scale - 0.5;
                let src_y = (dy as f32 + 0.5) * y_scale - 0.5;
                let x0 = (src_x.floor().max(0.0) as u32).min(w_max);
                let y0 = (src_y.floor().max(0.0) as u32).min(h_max);
                let x1 = (x0 + 1).min(w_max);
                let y1 = (y0 + 1).min(h_max);
                let fx = (src_x - src_x.floor()).clamp(0.0, 1.0);
                let fy = (src_y - src_y.floor()).clamp(0.0, 1.0);

                let sample_rgb = |sx: u32, sy: u32| -> (f32, f32, f32) {
                    let y_val = frame.y[(sy * frame.width + sx) as usize] as f32;
                    let (u_val, v_val) = chroma_sample(frame, sx, sy);
                    bt601_yuv_to_rgb(y_val, u_val, v_val)
                };

                let (r00, g00, b00) = sample_rgb(x0, y0);
                let (r10, g10, b10) = sample_rgb(x1, y0);
                let (r01, g01, b01) = sample_rgb(x0, y1);
                let (r11, g11, b11) = sample_rgb(x1, y1);

                let lerp = |a: f32, b: f32, c: f32, d: f32| -> f32 {
                    a * (1.0 - fx) * (1.0 - fy)
                        + b * fx * (1.0 - fy)
                        + c * (1.0 - fx) * fy
                        + d * fx * fy
                };
                let r = lerp(r00, r10, r01, r11) / 255.0;
                let g = lerp(g00, g10, g01, g11) / 255.0;
                let b = lerp(b00, b10, b01, b11) / 255.0;

                let o = (dy as usize) * sz + (dx as usize);
                self.rgb_chw_buf[o] = (r - MEAN[0]) / STD[0];
                self.rgb_chw_buf[plane + o] = (g - MEAN[1]) / STD[1];
                self.rgb_chw_buf[2 * plane + o] = (b - MEAN[2]) / STD[2];
            }
        }
    }

    fn detect_raw(
        &mut self,
        camera: CameraId,
        frame: &RawFrame<'_>,
    ) -> Result<Vec<Detection>, DetectorError> {
        reco_core::profile_scope!("detr_detect");

        {
            reco_core::profile_scope!("detr_preprocess");
            self.preprocess(frame);
        }

        let sz = self.input_size as usize;
        let input_tensor =
            TensorRef::from_array_view(([1, 3, sz, sz], self.rgb_chw_buf.as_slice()))
                .map_err(|e| DetectorError::InferenceFailed(format!("tensor build: {e}")))?;

        let outputs = {
            reco_core::profile_scope!("detr_inference");
            self.session
                .run(ort::inputs![input_tensor])
                .map_err(|e| DetectorError::InferenceFailed(format!("ort run: {e}")))?
        };

        let (dets_shape, dets_slice) = outputs["dets"]
            .try_extract_tensor::<f32>()
            .map_err(|e| DetectorError::InferenceFailed(format!("'dets' output extract: {e}")))?;
        let (labels_shape, labels_slice) = outputs["labels"]
            .try_extract_tensor::<f32>()
            .map_err(|e| DetectorError::InferenceFailed(format!("'labels' output extract: {e}")))?;

        let num_queries = dets_shape[1] as usize;
        let num_classes = labels_shape[2] as usize;

        let detections = postprocess_detr(
            dets_slice,
            labels_slice,
            num_queries,
            num_classes,
            camera,
            self.confidence_threshold,
        );
        drop(outputs);

        if !detections.is_empty() {
            log::debug!(
                "camera {:?}: {} ball detection(s) - {}",
                camera,
                detections.len(),
                detections
                    .iter()
                    .map(|d| format!(
                        "ball({:.0}%@{:.2},{:.2})",
                        d.confidence * 100.0,
                        d.center_x,
                        d.center_y
                    ))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }

        Ok(detections)
    }

    fn detect_preprocessed(
        &mut self,
        camera: CameraId,
        data: &[f32],
        input_size: u32,
        _src_width: u32,
        _src_height: u32,
    ) -> Result<Vec<Detection>, DetectorError> {
        reco_core::profile_scope!("detr_detect_preprocessed");

        let sz = input_size as usize;
        let expected = 3 * sz * sz;
        if data.len() != expected {
            return Err(DetectorError::InferenceFailed(format!(
                "PreprocessedChw: expected {expected} floats, got {}",
                data.len()
            )));
        }

        let input_tensor = TensorRef::from_array_view(([1, 3, sz, sz], data))
            .map_err(|e| DetectorError::InferenceFailed(format!("tensor build: {e}")))?;

        let outputs = {
            reco_core::profile_scope!("detr_inference");
            self.session
                .run(ort::inputs![input_tensor])
                .map_err(|e| DetectorError::InferenceFailed(format!("ort run: {e}")))?
        };

        let (dets_shape, dets_slice) = outputs["dets"]
            .try_extract_tensor::<f32>()
            .map_err(|e| DetectorError::InferenceFailed(format!("'dets' output extract: {e}")))?;
        let (labels_shape, labels_slice) = outputs["labels"]
            .try_extract_tensor::<f32>()
            .map_err(|e| DetectorError::InferenceFailed(format!("'labels' output extract: {e}")))?;

        let num_queries = dets_shape[1] as usize;
        let num_classes = labels_shape[2] as usize;

        let detections = postprocess_detr(
            dets_slice,
            labels_slice,
            num_queries,
            num_classes,
            camera,
            self.confidence_threshold,
        );
        drop(outputs);

        Ok(detections)
    }
}

impl UnifiedDetector for CpuDetrDetector {
    fn name(&self) -> &'static str {
        "ort-cpu-detr"
    }

    fn detect(
        &mut self,
        camera: CameraId,
        frame: &DetectorFrame<'_>,
    ) -> Result<Vec<Detection>, DetectorError> {
        match frame {
            DetectorFrame::Cpu(raw) => self.detect_raw(camera, raw),
            DetectorFrame::PreprocessedChw {
                data,
                input_size,
                src_width,
                src_height,
            } => self.detect_preprocessed(camera, data, *input_size, *src_width, *src_height),
            _ => Err(DetectorError::UnsupportedFrameKind),
        }
    }

    fn class_names(&self) -> Option<&[String]> {
        Some(&self.labels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile-time: `CpuDetrDetector` must satisfy the `UnifiedDetector`
    /// bounds so `StitchCore::set_detector` (`Box<dyn UnifiedDetector>`)
    /// accepts it, and stay `Send` (a regression here would mean a field
    /// accidentally holds shared mutable state or a raw pointer).
    #[test]
    fn cpu_detr_detector_is_unified_detector_send() {
        fn assert_unified_send<T: UnifiedDetector + Send + 'static>() {}
        assert_unified_send::<CpuDetrDetector>();
    }
}
