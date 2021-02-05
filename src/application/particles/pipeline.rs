use anyhow::{Context, Result};
use std::sync::Arc;
use vulkano::device::Device;
use vulkano::framebuffer::{RenderPassAbstract, Subpass};
use vulkano::impl_vertex;
use vulkano::pipeline::{
    viewport::Viewport, GraphicsPipeline, GraphicsPipelineAbstract,
};

type DynRenderPass = dyn RenderPassAbstract + Send + Sync;

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
        .line_width(1.0)
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
    //
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r#"
            #version 450
            #extension GL_ARB_separate_shader_objects : enable

            layout(location = 0) in vec2 pos;
            layout(location = 1) in vec4 color;

            layout(location = 0) out vec4 vertColor;

            void main() {
                vertColor = color;
                gl_PointSize = 64.0;
                gl_Position = vec4(pos, 0.0, 1.0);
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
