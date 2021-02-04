use anyhow::{Context, Result};
use std::sync::Arc;
use std::time::{Duration, Instant};
use vulkano::buffer::cpu_pool::CpuBufferPool;
use vulkano::command_buffer::{
    AutoCommandBuffer, AutoCommandBufferBuilder, DynamicState,
};
use vulkano::format::ClearValue;
use vulkano::pipeline::GraphicsPipelineAbstract;
use vulkano::swapchain::acquire_next_image;
use vulkano::sync::GpuFuture;
use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;

use crate::display::Display;

mod triangle_pipeline;

use triangle_pipeline::Vertex;

pub struct Application {
    // vulkan display resources
    display: Display,

    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,

    // vertex buffers
    buffer_pool: CpuBufferPool<[Vertex; 3]>,

    start: Instant,
}

impl Application {
    pub fn initialize() -> Result<Self> {
        let display =
            Display::create().context("unable to create the display")?;
        let pipeline = triangle_pipeline::create_graphics_pipeline(
            &display.device,
            display.swapchain.dimensions(),
            &display.render_pass,
        )?;
        let buffer_pool = CpuBufferPool::vertex_buffer(display.device.clone());

        Ok(Self {
            display,
            pipeline,
            buffer_pool,
            start: Instant::now(),
        })
    }

    fn build_command_buffer(
        &self,
        framebuffer_index: usize,
    ) -> AutoCommandBuffer {
        let family = self.display.graphics_queue.family();
        let framebuffer_image =
            &self.display.framebuffer_images[framebuffer_index];

        let vertices =
            Arc::new(self.buffer_pool.next(self.triangle_vertices()).unwrap());

        let mut builder = AutoCommandBufferBuilder::primary_simultaneous_use(
            self.display.device.clone(),
            family,
        )
        .unwrap();

        builder
            .begin_render_pass(
                framebuffer_image.clone(),
                vulkano::command_buffer::SubpassContents::Inline,
                vec![
                    ClearValue::Float([0.0, 0.0, 0.0, 1.0]),
                    ClearValue::Float([0.0, 0.0, 0.0, 1.0]),
                ],
            )
            .unwrap()
            .draw(
                self.pipeline.clone(),
                &DynamicState::none(),
                vec![vertices.clone()],
                (),
                (),
            )
            .unwrap()
            .end_render_pass()
            .unwrap();

        builder.build().unwrap()
    }

    fn triangle_vertices(&self) -> [Vertex; 3] {
        let time: Duration = Instant::now().duration_since(self.start);
        let t = time.as_secs_f32() / 10.0;
        let offset = (2.0 * 3.1415) / 3.0;

        [
            Vertex::new(
                [(offset * 0.0 + t).cos(), (offset * 0.0 + t).sin()],
                [1.0, 0.0, 0.0, 1.0],
            ),
            Vertex::new(
                [(offset * 1.0 + t).cos(), (offset * 1.0 + t).sin()],
                [1.0, 0.0, 0.0, 1.0],
            ),
            Vertex::new(
                [(offset * 2.0 + t).cos(), (offset * 2.0 + t).sin()],
                [1.0, 0.0, 0.0, 1.0],
            ),
        ]
    }

    /**
     * Render the screen.
     */
    fn render(&mut self) -> Result<()> {
        let (image_index, suboptimal, acquire_swapchain_future) =
            acquire_next_image(self.display.swapchain.clone(), None)
                .with_context(|| {
                    "unable to acquire next frame for rendering"
                })?;

        let command_buffer = self.build_command_buffer(image_index);

        let future = acquire_swapchain_future
            .then_execute(self.display.graphics_queue.clone(), command_buffer)
            .with_context(|| "unable to execute the display command buffer")?
            .then_swapchain_present(
                self.display.present_queue.clone(),
                self.display.swapchain.clone(),
                image_index,
            )
            .then_signal_fence_and_flush()
            .with_context(|| "unable to present, signal, and flush")?;

        // wait for the frame to finish
        future.wait(None).with_context(|| {
            "error while waiting for the frame to complete!"
        })?;

        if suboptimal {
            self.rebuild_swapchain_resources()?;
        }

        Ok(())
    }

    /// Rebuild the swapchain and command buffers
    fn rebuild_swapchain_resources(&mut self) -> Result<()> {
        log::debug!("rebuilding swapchain resources");
        self.display.rebuild_swapchain();
        self.pipeline = triangle_pipeline::create_graphics_pipeline(
            &self.display.device,
            self.display.swapchain.dimensions(),
            &self.display.render_pass,
        )
        .context("unable to rebuild the triangle pipeline")?;
        Ok(())
    }

    /**
     * Main application loop for this window. Blocks the thread until the
     * window is closed.
     */
    pub fn main_loop(mut self) -> Result<()> {
        let event_loop = self
            .display
            .event_loop
            .take()
            .context("unable to take ownership of the event loop")?;

        // render once before showing the window so it's not garbage
        self.render()
            .context("unable to render the first application frame")?;
        self.display.surface.window().set_visible(true);

        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Poll;

            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => {
                    *control_flow = ControlFlow::Exit;
                }

                Event::WindowEvent {
                    event: WindowEvent::Resized(_),
                    ..
                } => match self.rebuild_swapchain_resources() {
                    Err(error) => {
                        log::error!(
                            "unable to rebuild the swapchain {}",
                            error
                        );
                        *control_flow = ControlFlow::Exit;
                    }
                    Ok(_) => {}
                },

                Event::MainEventsCleared => match self.render() {
                    Err(error) => {
                        log::error!("unable to render the frame {}", error);
                        *control_flow = ControlFlow::Exit;
                    }
                    Ok(_) => {
                        self.display.surface.window().request_redraw();
                    }
                },

                _ => (),
            }
        });
    }
}
