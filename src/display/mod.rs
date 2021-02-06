use anyhow::{Context, Result};
use std::sync::Arc;
use vulkano::command_buffer::{AutoCommandBuffer, AutoCommandBufferBuilder};
use vulkano::device::{Device, Queue};
use vulkano::format::ClearValue;
use vulkano::framebuffer::{FramebufferAbstract, RenderPassAbstract};
use vulkano::image::swapchain::SwapchainImage;
use vulkano::instance::debug::DebugCallback;
use vulkano::instance::Instance;
use vulkano::swapchain::acquire_next_image;
use vulkano::swapchain::{Surface, Swapchain};
use vulkano::sync::GpuFuture;
use vulkano_win::VkSurfaceBuild;
use winit::dpi::LogicalSize;
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowBuilder};

mod device;
mod instance;
mod swapchain;

pub enum SwapchainState {
    Optimal,
    NeedsRebuild,
}

pub struct Display {
    // vulkan library resources
    pub instance: Arc<Instance>,
    pub debug_callback: Option<DebugCallback>,

    // window/surface resources
    pub surface: Arc<Surface<Window>>,
    pub event_loop: Option<EventLoop<()>>,
    pub render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    pub swapchain: Arc<Swapchain<Window>>,
    pub swapchain_images: Vec<Arc<SwapchainImage<Window>>>,
    pub framebuffer_images: Vec<Arc<dyn FramebufferAbstract + Send + Sync>>,

    // devices and queues
    pub device: Arc<Device>,
    pub graphics_queue: Arc<Queue>,
    pub present_queue: Arc<Queue>,
    pub compute_queue: Arc<Queue>,
}

impl Display {
    pub fn create() -> Result<Self> {
        let instance = instance::create_instance()
            .context("unable to create the vulkan instance")?;
        let debug_callback = instance::setup_debug_callback(&instance);

        let event_loop: EventLoop<()> = EventLoop::new();
        let surface = WindowBuilder::new()
            .with_title("vulkan starter")
            .with_resizable(true)
            .with_decorations(true)
            .with_visible(false)
            .with_inner_size(LogicalSize::new(1366, 768))
            .build_vk_surface(&event_loop, instance.clone())
            .context("unable to build the main vulkan window")?;

        let physical_device =
            device::pick_physical_device(&surface, &instance)?;

        let (device, graphics_queue, present_queue, compute_queue) =
            device::create_logical_device(&surface, &physical_device)?;
        let (swapchain, swapchain_images) = swapchain::create_swap_chain(
            &surface,
            &physical_device,
            &device,
            &graphics_queue,
            &present_queue,
        )?;

        let render_pass =
            swapchain::create_render_pass(&device, swapchain.format())?;

        let framebuffer_images = swapchain::create_framebuffers(
            &device,
            swapchain.format(),
            &swapchain_images,
            &render_pass,
        );

        Ok(Display {
            // library resources
            instance,
            debug_callback,

            // window/surface resources
            surface,
            event_loop: Option::Some(event_loop),
            render_pass,
            swapchain,
            swapchain_images,
            framebuffer_images,

            // devices and queues
            device,
            graphics_queue,
            present_queue,
            compute_queue,
        })
    }

    /// Rebuild the swapchain and dependent resources based on the the
    /// window's current size.
    pub fn rebuild_swapchain(&mut self) -> Result<()> {
        let size = self.surface.window().inner_size();
        let (swapchain, swapchain_images) = self
            .swapchain
            .recreate_with_dimensions([size.width, size.height])
            .context("unable to recreate the swapchain")?;
        let render_pass =
            swapchain::create_render_pass(&self.device, swapchain.format())
                .context("unable to recreate the render pass")?;
        let framebuffer_images = swapchain::create_framebuffers(
            &self.device,
            swapchain.format(),
            &swapchain_images,
            &render_pass,
        );

        self.swapchain = swapchain;
        self.swapchain_images = swapchain_images;
        self.render_pass = render_pass;
        self.framebuffer_images = framebuffer_images;

        Ok(())
    }

    /// Render the frame.
    ///
    /// @param graphics_queue_subbuffers a vector of secondary command buffers
    /// to be executed on the graphics queue
    pub fn render(
        &mut self,
        graphics_queue_subbuffers: Vec<AutoCommandBuffer>,
    ) -> Result<SwapchainState> {
        let (image_index, suboptimal, acquire_swapchain_future) =
            acquire_next_image(self.swapchain.clone(), None).with_context(
                || "unable to acquire next frame for rendering",
            )?;

        let render_buffer = self.build_render_pass_command_buffer(
            graphics_queue_subbuffers,
            image_index,
        )?;

        acquire_swapchain_future
            .then_execute(self.graphics_queue.clone(), render_buffer)
            .with_context(|| "unable to execute the display command buffer")?
            .then_swapchain_present(
                self.present_queue.clone(),
                self.swapchain.clone(),
                image_index,
            )
            .then_signal_fence_and_flush()
            .with_context(|| "unable to present, signal, and flush")?
            .wait(None)
            .with_context(|| "unable to complete the frame")?;

        if suboptimal {
            Ok(SwapchainState::NeedsRebuild)
        } else {
            Ok(SwapchainState::Optimal)
        }
    }

    /// Build a command buffer which renders the full render pass.
    ///
    /// Render passes are constructed by executing multiple subuffers.
    ///
    /// Unexpected behavior can occur if the graphics_queue_subbuffers share
    /// any data.
    fn build_render_pass_command_buffer(
        &self,
        graphics_queue_subbuffers: Vec<AutoCommandBuffer>,
        framebuffer_index: usize,
    ) -> Result<AutoCommandBuffer> {
        let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(
            self.device.clone(),
            self.graphics_queue.family(),
        )
        .with_context(|| "unable to create the command buffer builder")?;

        builder
            .begin_render_pass(
                self.framebuffer_images[framebuffer_index].clone(),
                vulkano::command_buffer::SubpassContents::SecondaryCommandBuffers,
                vec![
                    ClearValue::Float([0.0, 0.0, 0.0, 1.0]),
                    ClearValue::Float([0.0, 0.0, 0.0, 1.0]),
                ],
            )
            .with_context(|| "unable to begin the render pass")?;

        unsafe {
            // unsafe because vulkano does not check synchronization between
            // subbuffers and the main.
            builder
                .execute_commands_from_vec(graphics_queue_subbuffers)
                .with_context(|| {
                    "error while rendering secondary graphics commands"
                })?;
        }

        builder
            .end_render_pass()
            .with_context(|| "unable to end the render pass")?;

        builder
            .build()
            .with_context(|| "unable to build the command buffer")
    }
}
