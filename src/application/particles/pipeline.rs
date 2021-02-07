use anyhow::{Context, Result};
use std::sync::Arc;
use vulkano::{
    buffer::{BufferAccess, BufferUsage, ImmutableBuffer},
    descriptor::{
        descriptor_set::PersistentDescriptorSet, DescriptorSet,
        PipelineLayoutAbstract,
    },
    device::{Device, Queue},
    framebuffer::{RenderPassAbstract, Subpass},
    impl_vertex,
    pipeline::{
        vertex::{BufferlessDefinition, BufferlessVertices},
        viewport::Viewport,
        ComputePipeline, ComputePipelineAbstract, GraphicsPipeline,
        GraphicsPipelineAbstract,
    },
    sync::GpuFuture,
};

pub type ConcreteGraphicsPipeline = GraphicsPipeline<
    BufferlessDefinition,
    Box<dyn PipelineLayoutAbstract + Send + Sync>,
    Arc<dyn RenderPassAbstract + Send + Sync>,
>;
type DynRenderPass = dyn RenderPassAbstract + Send + Sync;
pub type Transform = vertex_shader::ty::Transform;

/// Create a transform descriptor set for the pipeline using the data in the
/// transform object.
pub fn create_transform_descriptor_set(
    pipeline: &Arc<ConcreteGraphicsPipeline>,
    graphics_queue: &Arc<Queue>,
    buffer: &Arc<dyn BufferAccess + Send + Sync>,
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
            .add_buffer(buffer.clone())
            .context("unable to bind the veretx buffer")?
            .build()
            .context("unable to build the persistent descriptor set")?,
    ))
}

pub fn create_compute_descriptor_set(
    pipeline: &Arc<dyn ComputePipelineAbstract + Send + Sync>,
    buffer: &Arc<dyn BufferAccess + Send + Sync>,
) -> Result<Arc<dyn DescriptorSet + Send + Sync>> {
    let layout = pipeline
        .descriptor_set_layout(0)
        .context("unable to get the compute pipeline's descriptor layout")?;
    Ok(Arc::new(
        PersistentDescriptorSet::start(layout.clone())
            .add_buffer(buffer.clone())
            .context("unable to bind the compute buffer")?
            .build()
            .context("unable to build the compute descriptor set")?,
    ))
}

pub fn create_graphics_pipeline(
    device: &Arc<Device>,
    swapchain_extent: [u32; 2],
    render_pass: &Arc<DynRenderPass>,
) -> Result<Arc<ConcreteGraphicsPipeline>> {
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
        .vertex_input(BufferlessDefinition {})
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

pub fn create_compute_pipeline(
    device: &Arc<Device>,
) -> Result<Arc<dyn ComputePipelineAbstract + Send + Sync>> {
    let compute = compute_shader::Shader::load(device.clone())
        .context("unable to load the compute shader")?;
    Ok(Arc::new(
        ComputePipeline::new(
            device.clone(),
            &compute.main_entry_point(),
            &(),
            None,
        )
        .context("unable to build the compute pipeline")?,
    ))
}

mod vertex_shader {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r#"
            #version 450
            #extension GL_ARB_separate_shader_objects : enable

            layout(location = 0) out vec4 vertColor;

            struct Vertex {
                vec2 pos;
                vec2 vel;
            };

            layout(set = 0, binding = 0) uniform Transform {
                mat4 projection;
            } ubo;

            layout(set = 0, binding = 1) buffer Data {
                Vertex vertices[];
            } data;

            void main() {
                Vertex vertex = data.vertices[gl_VertexIndex];
                vertColor = vec4(vertex.vel, 0.0, 1.0);
                gl_Position = ubo.projection * vec4(vertex.pos, 0.0, 1.0);
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

pub mod compute_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        types_meta: { #[derive(Copy, Clone, Default)] },
        src: r#"
        #version 450

        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

        struct Vertex {
            vec2 pos;
            vec2 vel;
        };

        layout(set = 0, binding = 0) buffer Data {
            Vertex vertices[];
        } data;

        void main() {
            uint idx = gl_GlobalInvocationID.x;
            Vertex vertex = data.vertices[idx];
            vertex.pos += vertex.vel * (1.0/60.0);
            data.vertices[idx] = vertex;
       }
        "#
    }
}
