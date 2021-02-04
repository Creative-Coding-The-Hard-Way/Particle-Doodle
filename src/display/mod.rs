use anyhow::{Context, Result};
use std::sync::Arc;
use vulkano::device::{Device, Queue};
use vulkano::framebuffer::{FramebufferAbstract, RenderPassAbstract};
use vulkano::image::swapchain::SwapchainImage;
use vulkano::instance::debug::DebugCallback;
use vulkano::instance::Instance;
use vulkano::swapchain::{Surface, Swapchain};
use vulkano_win::VkSurfaceBuild;
use winit::dpi::LogicalSize;
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowBuilder};

mod device;
mod instance;
mod swapchain;

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

        let (device, graphics_queue, present_queue) =
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
        })
    }

    /// Rebuild the swapchain and dependent resources based on the the
    /// window's current size.
    pub fn rebuild_swapchain(&mut self) {
        let size = self.surface.window().inner_size();
        let (swapchain, swapchain_images) = self
            .swapchain
            .recreate_with_dimensions([size.width, size.height])
            .expect("unable to recreate the swapchain!");
        let render_pass =
            swapchain::create_render_pass(&self.device, swapchain.format())
                .expect("unable to recreate the render pass");
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
    }
}
