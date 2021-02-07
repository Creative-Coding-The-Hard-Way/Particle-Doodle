mod pipeline;

use crate::display::Display;
use anyhow::{Context, Result};
use pipeline::Transform;
use rand::{thread_rng, Rng};
use std::sync::Arc;
use vulkano::{
    buffer::{BufferAccess, BufferUsage, ImmutableBuffer},
    command_buffer::{
        AutoCommandBuffer, AutoCommandBufferBuilder, DynamicState,
    },
    descriptor::descriptor_set::DescriptorSet,
    framebuffer::Subpass,
    pipeline::{vertex::BufferlessVertices, ComputePipelineAbstract},
    sync::GpuFuture,
};

type Mat4 = nalgebra::Matrix4<f32>;
pub type PushConstants = pipeline::PushConstants;

pub struct Particles {
    pipeline: Arc<pipeline::ConcreteGraphicsPipeline>,
    descriptor_set: Arc<dyn DescriptorSet + Send + Sync>,

    compute_pipeline: Arc<dyn ComputePipelineAbstract + Send + Sync>,
    compute_descriptor_set: Arc<dyn DescriptorSet + Send + Sync>,

    vertex_buffer: Arc<dyn BufferAccess + Send + Sync>,
}

impl Particles {
    pub fn new(display: &Display) -> Result<Self> {
        let pipeline = pipeline::create_graphics_pipeline(
            &display.device,
            display.swapchain.dimensions(),
            &display.render_pass,
        )?;

        let vertex_buffer = Self::initialize_vertices(display)?;

        let transform = Transform {
            projection: Mat4::identity().into(),
        };
        let descriptor_set = pipeline::create_transform_descriptor_set(
            &pipeline,
            &display.graphics_queue,
            &vertex_buffer,
            transform,
        )?;

        let compute_pipeline =
            pipeline::create_compute_pipeline(&display.device)?;
        let compute_descriptor_set = pipeline::create_compute_descriptor_set(
            &compute_pipeline,
            &vertex_buffer,
        )?;

        Ok(Self {
            pipeline,
            descriptor_set,
            compute_pipeline,
            compute_descriptor_set,
            vertex_buffer,
        })
    }

    pub fn reset_vertices(&mut self, display: &Display) -> Result<()> {
        self.vertex_buffer = Self::initialize_vertices(display)?;
        self.rebuild_swapchain_resources(display)?;
        self.compute_descriptor_set = pipeline::create_compute_descriptor_set(
            &self.compute_pipeline,
            &self.vertex_buffer,
        )?;
        Ok(())
    }

    fn initialize_vertices(
        display: &Display,
    ) -> Result<Arc<dyn BufferAccess + Send + Sync>> {
        let mut rng = thread_rng();
        let max = 262144 * 64;
        let step = 2.0 * std::f32::consts::PI / max as f32;
        let vertices = (0..max).map(|i| {
            let radius = rng.gen_range(0.2..1.0);
            let angle = i as f32 * step;
            pipeline::compute_shader::ty::Vertex {
                pos: [radius * angle.cos(), radius * angle.sin()],
                vel: [0.0, 0.0],
                ..Default::default()
            }
        });

        let (buffer, future) = ImmutableBuffer::from_iter(
            vertices,
            BufferUsage::all(),
            display.compute_queue.clone(),
        )
        .context("unable to build vertex buffer for compute")?;
        future
            .then_signal_fence_and_flush()
            .context("unable to upload vertex data for initialization")?
            .wait(None)
            .context(
                "interruped while waiting for vertex upload to complete",
            )?;

        Ok(Arc::new(buffer))
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

        const WORLD_SIZE: f32 = 2.0;
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
            &self.vertex_buffer,
            transform,
        )?;

        Ok(())
    }

    pub fn tick(
        &self,
        display: &Display,
        push_constants: PushConstants,
    ) -> Result<()> {
        let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(
            display.device.clone(),
            display.compute_queue.family(),
        )
        .with_context(|| {
            "unable to create the compute command buffer builder"
        })?;
        builder
            .dispatch(
                [262144, 1, 1],
                self.compute_pipeline.clone(),
                self.compute_descriptor_set.clone(),
                push_constants,
            )
            .with_context(|| "unable to dispatch the compute pipeline")?;
        let commands = builder
            .build()
            .with_context(|| "unable to build the comput command buffer")?;

        vulkano::sync::now(display.device.clone())
            .then_execute(display.compute_queue.clone(), commands)
            .with_context(|| "unable to execute compute commands")?
            .then_signal_fence_and_flush()
            .with_context(|| {
                "error while waiting for the compute pipeline to execute"
            })?
            .wait(None)
            .with_context(|| {
                "error while waiting for the cpu to be notified"
            })?;

        Ok(())
    }

    pub fn draw(&self, display: &Display) -> Result<AutoCommandBuffer> {
        let mut builder =
            AutoCommandBufferBuilder::secondary_graphics_one_time_submit(
                display.device.clone(),
                display.graphics_queue.family(),
                Subpass::from(display.render_pass.clone(), 0).with_context(
                    || "unable to select subpass for particles",
                )?,
            )
            .with_context(|| "unable to create the command buffer builder")?;
        let vertices = BufferlessVertices {
            vertices: 262144 * 64,
            instances: 1,
        };
        builder
            .draw(
                self.pipeline.clone(),
                &DynamicState::none(),
                vertices,
                vec![self.descriptor_set.clone()],
                (),
            )
            .with_context(|| "unable to issue draw command")?;
        builder
            .build()
            .with_context(|| "unable to build the command buffer")
    }
}
