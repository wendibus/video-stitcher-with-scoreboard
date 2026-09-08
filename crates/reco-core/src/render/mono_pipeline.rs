//! Single-input cylindrical stitch pipeline.
//!
//! Mirrors [`super::pipeline::StitchPipeline`] but for one camera and
//! [`CylindricalProjectionConfig`] instead of [`MatchCalibration`] +
//! `SceneGeometry` - see [`crate::projection::CylindricalProjection`]'s
//! module docs for why this is a separate type rather than an `Option`
//! threaded through `StitchPipeline`: the two pipelines share almost no
//! GPU state (one camera plane vs. two, a raycasting shader vs. a
//! textured-quad one), so keeping them apart avoids `Option`-checking
//! every stereo-only field on every call.
//!
//! Used by [`crate::core::mono::MonoStitchCore`] the same way
//! `StitchPipeline` is used by [`crate::core::StitchCore`].

use super::mono_renderer::MonoRenderer;
use super::pipeline::PipelineError;
use super::planes::YuvPlanes;
use super::renderer::InputFormat;
use super::viewport::ViewportConfig;
use crate::calibration::MAX_DIM;
use crate::gpu::GpuContext;
use crate::projection::CylindricalProjectionConfig;

/// The single-input cylindrical stitch pipeline.
///
/// Owns the GPU context, the cylindrical projection config, and the
/// [`MonoRenderer`]. Consumers provide YUV420P frames and receive a
/// rendered RGBA command buffer via [`Self::render_to_target`].
pub struct MonoPipeline {
    gpu: GpuContext,
    config: CylindricalProjectionConfig,
    viewport: ViewportConfig,
    renderer: MonoRenderer,
    input_width: u32,
    input_height: u32,
}

impl MonoPipeline {
    /// Create a mono pipeline with an existing GPU context.
    pub fn with_gpu(
        gpu: GpuContext,
        config: CylindricalProjectionConfig,
        viewport: ViewportConfig,
        input_width: u32,
        input_height: u32,
        output_format: impl Into<wgpu::TextureFormat>,
        input_format: InputFormat,
    ) -> Result<Self, PipelineError> {
        if let Err(e) = viewport.validate() {
            return Err(PipelineError::InvalidConfig { reason: e });
        }
        if input_width == 0 || input_height == 0 {
            return Err(PipelineError::InvalidConfig {
                reason: format!("input dimensions must be > 0, got {input_width}x{input_height}"),
            });
        }
        if input_width > MAX_DIM || input_height > MAX_DIM {
            return Err(PipelineError::InvalidConfig {
                reason: format!(
                    "input dimensions {input_width}x{input_height} exceed MAX_DIM ({MAX_DIM})"
                ),
            });
        }

        let output_format = output_format.into();
        let renderer = MonoRenderer::new(
            &gpu,
            viewport.width,
            viewport.height,
            input_width,
            input_height,
            output_format,
            input_format,
        );

        log::info!(
            "Mono pipeline initialized: {}x{} output, GPU: {}, sweep={:.0}deg",
            viewport.width,
            viewport.height,
            gpu.adapter_info.name,
            config.angular_sweep_rad.to_degrees(),
        );

        Ok(Self {
            gpu,
            config,
            viewport,
            renderer,
            input_width,
            input_height,
        })
    }

    /// The name of the GPU this pipeline is running on.
    pub fn gpu_name(&self) -> &str {
        self.gpu.gpu_name()
    }

    /// Shared reference to the GPU context.
    pub fn gpu(&self) -> &GpuContext {
        &self.gpu
    }

    /// The cylindrical projection config this pipeline was created with.
    pub fn projection_config(&self) -> &CylindricalProjectionConfig {
        &self.config
    }

    /// The current output viewport configuration.
    pub fn viewport(&self) -> &ViewportConfig {
        &self.viewport
    }

    /// Input frame dimensions as `(width, height)`.
    pub fn source_info(&self) -> (u32, u32) {
        (self.input_width, self.input_height)
    }

    /// Access the internal render target texture.
    pub fn render_target(&self) -> &wgpu::Texture {
        self.renderer.render_target()
    }

    /// Upload a YUV420P frame and render the current pose to the
    /// internal target.
    pub fn render_to_target(
        &self,
        frame: &YuvPlanes<'_>,
        yaw: f32,
        pitch: f32,
    ) -> Result<wgpu::CommandBuffer, PipelineError> {
        self.renderer
            .upload_yuv(&self.gpu, frame.y, frame.u, frame.v)?;
        Ok(self.renderer.render_to_target(
            &self.gpu,
            &self.config,
            yaw,
            pitch,
            self.viewport.rig_tilt,
            self.viewport.rig_roll,
            self.viewport.fov_degrees,
        ))
    }
}
