//! GPU renderer for the single-input cylindrical mono pipeline.
//!
//! Mirrors [`super::renderer::Renderer`] but for one camera plane and the
//! cylindrical composite shader (`shaders/cylindrical_mono.wgsl`) instead
//! of the fisheye L-shape one - see
//! [`crate::projection::CylindricalProjection`] for the projection
//! geometry this renders. Reuses `Renderer`'s plane/texture setup and
//! YUV upload helpers (`pub(crate)` for exactly this) rather than
//! duplicating them; only the shader, bind-group-1 uniform layout, and
//! render pass are different (single draw call, no vertex buffer - the
//! shader generates a full-screen triangle from `vertex_index` alone).

use super::renderer::{
    InputFormat, PlaneResources, RenderError, Renderer, matrix4_to_columns, upload_yuv, view_matrix,
};
use crate::gpu::GpuContext;
use crate::projection::{CYLINDRICAL_CAMERA_REST_POSITION, CylindricalProjectionConfig};

use bytemuck::{Pod, Zeroable};

/// Uniform buffer layout - must match `CylUniforms` in
/// `cylindrical_mono.wgsl` exactly (96 bytes: a `mat4x4<f32>` plus 8
/// `f32`s, already 16-byte aligned).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct CylUniformsGpu {
    view: [[f32; 4]; 4],
    focal_length: f32,
    angular_sweep: f32,
    screen_rotation: f32,
    video_height: f32,
    v_fov: f32,
    aspect: f32,
    is_nv12: f32,
    _pad0: f32,
}

/// The GPU renderer for the single-input cylindrical projection.
///
/// Holds all wgpu resources for one camera plane: pipeline, textures,
/// bind groups. Created once per [`super::mono_pipeline::MonoPipeline`]
/// and reused for every frame.
pub(crate) struct MonoRenderer {
    pipeline: wgpu::RenderPipeline,
    plane: PlaneResources,
    render_target: wgpu::Texture,
    render_target_view: wgpu::TextureView,
    output_width: u32,
    output_height: u32,
    input_format: InputFormat,
}

impl MonoRenderer {
    /// Create a new mono renderer with all GPU resources.
    pub(crate) fn new(
        gpu: &GpuContext,
        output_width: u32,
        output_height: u32,
        input_width: u32,
        input_height: u32,
        output_format: wgpu::TextureFormat,
        input_format: InputFormat,
    ) -> Self {
        let device = &gpu.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cylindrical_mono"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/cylindrical_mono.wgsl").into(),
            ),
        });

        // Same texture bind group layout shape as the stereo `Renderer`
        // (Y/U/V planes + sampler) - `cylindrical_mono.wgsl`'s `@group(0)`
        // bindings mirror `fisheye.wgsl`'s exactly so the two share this
        // layout convention (and `Renderer::create_plane_resources` below).
        let texture_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mono_texture_layout"),
            entries: &[
                texture_entry(0),
                texture_entry(1),
                texture_entry(2),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mono_uniform_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mono_pipeline_layout"),
            bind_group_layouts: &[&texture_layout, &uniform_layout],
            immediate_size: 0,
        });

        // No vertex buffer: `vs_fullscreen` builds a full-screen triangle
        // from `@builtin(vertex_index)` alone (single camera, no per-plane
        // 3D placement, so there is nothing for a vertex buffer to carry).
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mono_render_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_fullscreen"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_cylindrical_mono"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: output_format,
                    // No seam to blend - a single camera fills the whole
                    // frame, so plain overwrite is correct and cheaper
                    // than the stereo path's alpha blend.
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mono_video_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let plane = Renderer::create_plane_resources(
            device,
            &texture_layout,
            &uniform_layout,
            &sampler,
            input_width,
            input_height,
            input_format,
            "mono",
        );

        let render_target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mono_render_target"),
            size: wgpu::Extent3d {
                width: output_width,
                height: output_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: output_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let render_target_view = render_target.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            pipeline,
            plane,
            render_target,
            render_target_view,
            output_width,
            output_height,
            input_format,
        }
    }

    /// Upload YUV420P planes to the single camera plane's textures.
    pub(crate) fn upload_yuv(
        &self,
        gpu: &GpuContext,
        y: &[u8],
        u: &[u8],
        v: &[u8],
    ) -> Result<(), RenderError> {
        upload_yuv(gpu, &self.plane, y, u, v)
    }

    /// Access the internal render target texture.
    pub(crate) fn render_target(&self) -> &wgpu::Texture {
        &self.render_target
    }

    /// Render the cylindrical composite to the internal target.
    ///
    /// `yaw`/`pitch` position the virtual camera on the cylinder axis;
    /// `rig_tilt`/`rig_roll` correct for a physically un-level mount
    /// (same convention as the stereo path's `view_matrix`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn render_to_target(
        &self,
        gpu: &GpuContext,
        config: &CylindricalProjectionConfig,
        yaw: f32,
        pitch: f32,
        rig_tilt: f32,
        rig_roll: f32,
        fov_degrees: f32,
    ) -> wgpu::CommandBuffer {
        // `view_matrix` returns the world-to-view (look-at) rotation used
        // to transform world points into camera space. The cylindrical
        // shader instead rotates a camera-space ray *into* world space
        // (`ray_w = u.view * ray`), so it needs the inverse - which, for
        // a pure rotation (camera at the origin, no translation), is
        // just the transpose.
        let world_to_view = view_matrix(
            &CYLINDRICAL_CAMERA_REST_POSITION,
            yaw,
            pitch,
            rig_tilt,
            rig_roll,
        );
        let view_to_world = world_to_view.transpose();

        let aspect = self.output_width as f32 / self.output_height as f32;
        let uniforms = CylUniformsGpu {
            view: matrix4_to_columns(&view_to_world),
            focal_length: config.focal_length,
            angular_sweep: config.angular_sweep_rad,
            screen_rotation: config.screen_rotation_rad,
            video_height: config.video_height,
            v_fov: fov_degrees.to_radians(),
            aspect,
            is_nv12: match self.input_format {
                InputFormat::Nv12 => 1.0,
                InputFormat::Yuv420p | InputFormat::Bgra => 0.0,
            },
            _pad0: 0.0,
        };
        gpu.queue
            .write_buffer(&self.plane.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mono_stitch_to_target"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mono_stitch_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.render_target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.plane.texture_bind_group, &[]);
            pass.set_bind_group(1, &self.plane.uniform_bind_group, &[]);
            // Full-screen triangle: 3 vertices, no vertex buffer.
            pass.draw(0..3, 0..1);
        }
        encoder.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The uniform struct's size/alignment must match what the WGSL
    /// `CylUniforms` struct expects (96 bytes, 16-byte aligned) - a
    /// mismatch here would silently corrupt the shader's view of the
    /// buffer instead of failing loudly.
    #[test]
    fn cyl_uniforms_gpu_matches_wgsl_layout() {
        assert_eq!(std::mem::size_of::<CylUniformsGpu>(), 96);
        assert_eq!(std::mem::size_of::<CylUniformsGpu>() % 16, 0);
    }
}
