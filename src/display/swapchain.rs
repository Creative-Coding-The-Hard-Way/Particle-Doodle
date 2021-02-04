use anyhow::{Context, Result};
use log;
use std::cmp::{max, min};
use std::sync::Arc;
use vulkano::device::{Device, Queue};
use vulkano::format::Format;
use vulkano::framebuffer::{
    Framebuffer, FramebufferAbstract, RenderPassAbstract,
};
use vulkano::image::AttachmentImage;
use vulkano::image::{swapchain::SwapchainImage, ImageUsage};
use vulkano::instance::PhysicalDevice;
use vulkano::single_pass_renderpass;
use vulkano::swapchain::{
    Capabilities, ColorSpace, CompositeAlpha, FullscreenExclusive, PresentMode,
    Surface, Swapchain,
};
use vulkano::sync::SharingMode;
use winit::window::Window;

type DynRenderPass = dyn RenderPassAbstract + Send + Sync;

/// Build a render pass which uses the maximum multisampling level supported
/// by this device.
pub fn create_render_pass(
    device: &Arc<Device>,
    color_format: Format,
) -> Result<Arc<DynRenderPass>> {
    let samples = pick_sample_count(&device.physical_device());
    log::debug!("framebuffer samples {}", samples);

    let render_pass = single_pass_renderpass!(
        device.clone(),
        attachments: {
            intermediary: {
                load: Clear,
                store: DontCare,
                format: color_format,
                samples: samples,
            },

            color: {
                load: Clear,
                store: Store,
                format: color_format,
                samples: 1,
            }
        },
        pass: {
            color: [intermediary],
            depth_stencil: {}
            resolve: [color]
        }
    )
    .context("unable to create renderpass")?;

    Ok(Arc::new(render_pass))
}

/// Pick the largest supported sampling count for this device
fn pick_sample_count(physical_device: &PhysicalDevice) -> u32 {
    let counts = physical_device.limits().framebuffer_color_sample_counts();
    [
        (vk_sys::SAMPLE_COUNT_64_BIT, 64),
        (vk_sys::SAMPLE_COUNT_32_BIT, 32),
        (vk_sys::SAMPLE_COUNT_16_BIT, 16),
        (vk_sys::SAMPLE_COUNT_8_BIT, 8),
        (vk_sys::SAMPLE_COUNT_4_BIT, 4),
        (vk_sys::SAMPLE_COUNT_2_BIT, 2),
    ]
    .iter()
    .find(|(mask, _)| counts & *mask > 0)
    .map(|(_, samples)| *samples)
    .unwrap_or(1)
}

pub fn create_framebuffers(
    device: &Arc<Device>,
    color_format: Format,
    swapchain_images: &[Arc<SwapchainImage<Window>>],
    render_pass: &Arc<dyn RenderPassAbstract + Send + Sync>,
) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
    swapchain_images
        .iter()
        .map(|image| {
            let intermediary = AttachmentImage::transient_multisampled(
                device.clone(),
                image.dimensions(),
                render_pass.num_samples(0).unwrap(),
                color_format,
            )
            .unwrap();
            let fba: Arc<dyn FramebufferAbstract + Send + Sync> = Arc::new(
                Framebuffer::start(render_pass.clone())
                    .add(intermediary.clone())
                    .unwrap()
                    .add(image.clone())
                    .unwrap()
                    .build()
                    .unwrap(),
            );
            fba
        })
        .collect::<Vec<_>>()
}

/// Construct a swapchain and it's owned images
pub fn create_swap_chain(
    surface: &Arc<Surface<Window>>,
    physical_device: &PhysicalDevice,
    logical_device: &Arc<Device>,
    graphics_queue: &Arc<Queue>,
    present_queue: &Arc<Queue>,
) -> Result<(Arc<Swapchain<Window>>, Vec<Arc<SwapchainImage<Window>>>)> {
    let capabilities = surface.capabilities(*physical_device)?;
    let swap_format = choose_swap_surface_format(&capabilities);
    let swap_present_mode = choose_swap_present_mode(&capabilities);
    let swap_extent = choose_swap_extent(surface, &capabilities);
    let swap_image_count = choose_image_count(&capabilities);
    let sharing_mode = choose_sharing_mode(graphics_queue, present_queue);

    let image_usage = ImageUsage {
        color_attachment: true,
        ..ImageUsage::none()
    };

    let (swapchain, images) = Swapchain::new(
        logical_device.clone(),
        surface.clone(),
        swap_image_count,
        swap_format.0,
        swap_extent,
        1,
        image_usage,
        sharing_mode,
        capabilities.current_transform,
        CompositeAlpha::Opaque,
        swap_present_mode,
        FullscreenExclusive::AppControlled,
        false,
        swap_format.1,
    )
    .context("unable to build swapchain")?;

    Ok((swapchain, images))
}

fn choose_sharing_mode(
    graphics_queue: &Arc<Queue>,
    present_queue: &Arc<Queue>,
) -> SharingMode {
    let same_queue =
        graphics_queue.id_within_family() == present_queue.id_within_family();
    if same_queue {
        SharingMode::Exclusive
    } else {
        SharingMode::Concurrent(vec![
            graphics_queue.id_within_family(),
            present_queue.id_within_family(),
        ])
    }
}

fn choose_image_count(capabilities: &Capabilities) -> u32 {
    let suggested_count = capabilities.min_image_count + 1;
    if let Some(max_count) = capabilities.max_image_count {
        min(suggested_count, max_count)
    } else {
        suggested_count
    }
}

/// Select a format and color space from the available formats
fn choose_swap_surface_format(
    capabilities: &Capabilities,
) -> (Format, ColorSpace) {
    log::info!("display formats {:?}", capabilities.supported_formats);

    let (format, color_space) = *capabilities
        .supported_formats
        .iter()
        .find(|(format, color_space)| {
            *format == Format::B8G8R8A8Srgb
                && *color_space == ColorSpace::SrgbNonLinear
        })
        .unwrap_or_else(|| &capabilities.supported_formats[0]);

    log::info!("chosen format: {:?}", (format, color_space));

    (format, color_space)
}

/// Select the presentation mode
fn choose_swap_present_mode(capabilities: &Capabilities) -> PresentMode {
    let mode = if capabilities.present_modes.mailbox {
        PresentMode::Mailbox
    } else {
        PresentMode::Fifo
    };
    log::info!("selected presentation mode: {:?}", mode);
    mode
}

/// Select the swapchain presentation extent.
/// Some window managers will automatically fill the current_extent property.
/// Otherwise, an extent will need to be decided by hand.
fn choose_swap_extent(
    surface: &Arc<Surface<Window>>,
    capabilities: &Capabilities,
) -> [u32; 2] {
    // if an extent already exists, just use it
    if let Some(extent) = capabilities.current_extent {
        extent
    } else {
        let physical_size = surface.window().inner_size();
        let width = clamp(
            physical_size.width,
            capabilities.min_image_extent[0],
            capabilities.max_image_extent[0],
        );
        let height = clamp(
            physical_size.height,
            capabilities.min_image_extent[1],
            capabilities.max_image_extent[1],
        );
        [width, height]
    }
}

fn clamp(x: u32, lower: u32, upper: u32) -> u32 {
    max(lower, min(x, upper))
}
