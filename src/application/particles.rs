use crate::display::Display;
use anyhow::{Context, Result};
use std::sync::Arc;
use vulkano::command_buffer::{
    AutoCommandBuffer, AutoCommandBufferBuilder, DynamicState,
};
use vulkano::pipeline::GraphicsPipelineAbstract;
use vulkano::{buffer::cpu_pool::CpuBufferPool, framebuffer::Subpass};

mod pipeline;

pub type Vertex = pipeline::Vertex;

pub struct Particles {
    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,

    // vertex buffers
    vertex_buffer_pool: CpuBufferPool<Vertex>,

    // vertices
    pub vertices: Vec<Vertex>,
}

impl Particles {
    pub fn new(display: &Display) -> Result<Self> {
        let pipeline = pipeline::create_graphics_pipeline(
            &display.device,
            display.swapchain.dimensions(),
            &display.render_pass,
        )?;

        let vertex_buffer_pool =
            CpuBufferPool::vertex_buffer(display.device.clone());

        Ok(Self {
            pipeline,
            vertex_buffer_pool,
            vertices: vec![
                Vertex::new([0.0, -0.5], [1.0, 1.0, 1.0, 1.0]),
                Vertex::new([0.5, 0.5], [0.0, 0.0, 1.0, 1.0]),
                Vertex::new([-0.5, 0.5], [0.0, 1.0, 0.0, 1.0]),
            ],
        })
    }

    pub fn rebuild_swapchain_resources(
        &mut self,
        display: &Display,
    ) -> Result<()> {
        self.pipeline = pipeline::create_graphics_pipeline(
            &display.device,
            display.swapchain.dimensions(),
            &display.render_pass,
        )?;
        Ok(())
    }

    pub fn draw(&self, display: &Display) -> Result<AutoCommandBuffer> {
        let vertex_buffer = Arc::new(
            self.vertex_buffer_pool
                .chunk(self.vertices.iter().cloned())?,
        );

        let mut builder =
            AutoCommandBufferBuilder::secondary_graphics_one_time_submit(
                display.device.clone(),
                display.graphics_queue.family(),
                Subpass::from(display.render_pass.clone(), 0).with_context(
                    || "unable to select subpass for particles",
                )?,
            )
            .with_context(|| "unable to create the command buffer builder")?;
        builder
            .draw(
                self.pipeline.clone(),
                &DynamicState::none(),
                vec![vertex_buffer],
                (),
                (),
            )
            .with_context(|| "unable to issue draw command")?;
        builder
            .build()
            .with_context(|| "unable to build the command buffer")
    }
}
