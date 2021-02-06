mod pipeline;

use crate::display::Display;
use anyhow::{Context, Result};
use pipeline::Transform;
use std::sync::Arc;
use vulkano::{
    buffer::cpu_pool::CpuBufferPool,
    command_buffer::{
        AutoCommandBuffer, AutoCommandBufferBuilder, DynamicState,
    },
    descriptor::descriptor_set::DescriptorSet,
    framebuffer::Subpass,
    pipeline::GraphicsPipelineAbstract,
};

type Mat4 = nalgebra::Matrix4<f32>;
pub type Vertex = pipeline::Vertex;

pub struct Particles {
    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,

    // vertex buffers
    vertex_buffer_pool: CpuBufferPool<Vertex>,

    // pipeline descriptor for the screen transform
    descriptor_set: Arc<dyn DescriptorSet + Send + Sync>,

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

        let transform = Transform {
            projection: Mat4::identity().into(),
        };
        let descriptor_set = pipeline::create_transform_descriptor_set(
            &pipeline,
            &display.graphics_queue,
            transform,
        )?;

        Ok(Self {
            pipeline,
            vertex_buffer_pool,
            descriptor_set,
            vertices: vec![],
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

        const WORLD_SIZE: f32 = 4.0;
        let [width, height] = display.swapchain.dimensions();
        let aspect = width as f32 / height as f32;
        let world_width = aspect * WORLD_SIZE;
        let transform = Transform {
            projection: Mat4::new_orthographic(
                -world_width / 2.0,
                world_width / 2.0,
                WORLD_SIZE / 2.0,
                -WORLD_SIZE / 2.0,
                1.0,
                -1.0,
            )
            .into(),
        };
        self.descriptor_set = pipeline::create_transform_descriptor_set(
            &self.pipeline,
            &display.graphics_queue,
            transform,
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
                self.descriptor_set.clone(),
                (),
            )
            .with_context(|| "unable to issue draw command")?;
        builder
            .build()
            .with_context(|| "unable to build the command buffer")
    }
}
