use anyhow::{Context, Result};
use std::sync::Arc;
use vulkano::{
    buffer::{BufferUsage, ImmutableBuffer},
    descriptor::{descriptor_set::PersistentDescriptorSet, DescriptorSet},
    device::{Device, Queue},
    framebuffer::{RenderPassAbstract, Subpass},
    impl_vertex,
    pipeline::{
        viewport::Viewport, GraphicsPipeline, GraphicsPipelineAbstract,
    },
    sync::GpuFuture,
};

type DynRenderPass = dyn RenderPassAbstract + Send + Sync;
pub type Transform = vertex_shader::ty::Transform;

#[derive(Default, Debug, Copy, Clone)]
pub struct Vertex {
    pub pos: [f32; 2],
    pub color: [f32; 4],
}

impl_vertex!(Vertex, pos, color);

impl Vertex {
    pub fn new(pos: [f32; 2], color: [f32; 4]) -> Self {
        Self { pos, color }
    }
}

/// Create a transform descriptor set for the pipeline using the data in the
/// transform object.
pub fn create_transform_descriptor_set(
    pipeline: &Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    graphics_queue: &Arc<Queue>,
    transform: Transform,
) -> Result<Arc<dyn DescriptorSet + Send + Sync>> {
    let (uniform_buffer, future) = ImmutableBuffer::from_data(
        transform,
        BufferUsage::uniform_buffer(),
        graphics_queue.clone(),
    )
    .context("unable to create the uniform buffer")?;
    future
        .then_signal_fence_and_flush() // wait for the GPU to finish the operation
        .context("error while waiting for uniform buffer to be written")?
        .wait(None) // wait for the CPU to hear about it
        .context("uniform buffer timeout")?;

    let layout = pipeline
        .descriptor_set_layout(0)
        .context("unable to get the pipeline's transform descriptor set")?;
    Ok(Arc::new(
        PersistentDescriptorSet::start(layout.clone())
            .add_buffer(uniform_buffer)
            .context("unable to bind the transform buffer")?
            .build()
            .context("unable to build the persistent descriptor set")?,
    ))
}

pub fn create_graphics_pipeline(
    device: &Arc<Device>,
    swapchain_extent: [u32; 2],
    render_pass: &Arc<DynRenderPass>,
) -> Result<Arc<dyn GraphicsPipelineAbstract + Send + Sync>> {
    let vert = vertex_shader::Shader::load(device.clone())
        .context("unable to load the vertex shader")?;
    let frag = fragment_shader::Shader::load(device.clone())
        .context("unable to load the fragment shader")?;

    let dimensions = [swapchain_extent[0] as f32, swapchain_extent[1] as f32];
    let viewport = Viewport {
        dimensions,
        origin: [0.0, 0.0],
        depth_range: 0.0..1.0,
    };

    let pipeline = GraphicsPipeline::start()
        .vertex_input_single_buffer::<Vertex>()
        .vertex_shader(vert.main_entry_point(), ())
        .fragment_shader(frag.main_entry_point(), ())
        .viewports(vec![viewport])
        .depth_clamp(false)
        .polygon_mode_fill()
        .depth_write(false)
        .sample_shading_disabled()
        .blend_alpha_blending()
        .point_list()
        .render_pass(
            Subpass::from(render_pass.clone(), 0)
                .context("could not create the pipeline subpass")?,
        )
        .build(device.clone())
        .context("could not create the graphics pipeline")?;

    Ok(Arc::new(pipeline))
}

mod vertex_shader {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r#"
            #version 450
            #extension GL_ARB_separate_shader_objects : enable

            layout(location = 0) in vec2 pos;
            layout(location = 1) in vec4 color;

            layout(location = 0) out vec4 vertColor;

            layout(set = 0, binding = 0) uniform Transform {
                mat4 projection;
            } ubo;

            void main() {
                vertColor = color;
                gl_PointSize = 10.0;
                gl_Position = ubo.projection * vec4(pos, 0.0, 1.0);
            }
            "#
    }
}

mod fragment_shader {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r#"
            #version 450
            #extension GL_ARB_separate_shader_objects : enable

            layout(location = 0) in vec4 fragColor;
            layout(location = 0) out vec4 outColor;

            void main() {
               outColor = fragColor;
            }
            "#
    }
}
