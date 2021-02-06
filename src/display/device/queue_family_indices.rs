use anyhow::{Context, Result};
use std::sync::Arc;
use vulkano::device::{Queue, QueuesIter};
use vulkano::instance::PhysicalDevice;
use vulkano::swapchain::Surface;
use winit::window::Window;

pub struct QueueFamilyIndices {
    graphics_family: usize,
    present_family: usize,
    compute_family: usize,
}

impl QueueFamilyIndices {
    /// Find the queue family indices for the given device
    pub fn find(
        surface: &Arc<Surface<Window>>,
        device: &PhysicalDevice,
    ) -> Result<Self> {
        let mut graphics = None;
        let mut present = None;
        let mut compute = None;

        for (i, family) in device.queue_families().enumerate() {
            if family.supports_graphics() {
                graphics = Some(i);
            }

            if surface.is_supported(family)? {
                present = Some(i);
            }

            if family.supports_compute() {
                compute = Some(i);
            }

            if graphics.is_some() && present.is_some() && compute.is_some() {
                break;
            }
        }

        graphics
            .zip(present)
            .zip(compute)
            .map(|((graphics_family, present_family), compute_family)| Self {
                graphics_family,
                present_family,
                compute_family,
            })
            .context("unable to find all required queue families for this physical device")
    }

    /// Return the set of unique queue family indices
    pub fn unique_indices(&self) -> Vec<usize> {
        let mut indices = vec![self.graphics_family];
        if !indices.contains(&self.present_family) {
            indices.push(self.present_family);
        }
        if !indices.contains(&self.compute_family) {
            indices.push(self.compute_family);
        }
        indices
    }

    /// get the graphics and present queues based on the index order returned
    /// by unique_indices
    pub fn take_queues(
        &self,
        mut queues: QueuesIter,
    ) -> Result<(Arc<Queue>, Arc<Queue>, Arc<Queue>)> {
        let graphics_queue = queues
            .next()
            .context("could not find a graphics queue for this device")?;

        let present_queue = if self.is_same_queue() {
            graphics_queue.clone()
        } else {
            queues.next().context(
                "could not find a presentation queue for this device",
            )?
        };

        let compute_queue = queues
            .next()
            .or_else(|| {
                if graphics_queue.family().supports_compute() {
                    Some(graphics_queue.clone())
                } else if present_queue.family().supports_compute() {
                    Some(present_queue.clone())
                } else {
                    None
                }
            })
            .context("unable to construct a compute queue for this device")?;

        Ok((graphics_queue, present_queue, compute_queue))
    }

    fn is_same_queue(&self) -> bool {
        self.graphics_family == self.present_family
    }
}
