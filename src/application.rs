mod particles;

use crate::display::Display;
use anyhow::{Context, Result};
use particles::Particles;
use std::f32::consts::PI;
use std::time::Instant;
use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;

pub struct Application {
    display: Display,
    particles: Particles,
    start: Instant,
}

impl Application {
    pub fn initialize() -> Result<Self> {
        let display =
            Display::create().context("unable to create the display")?;
        let particles = Particles::new(&display)?;

        Ok(Self {
            display,
            particles,
            start: Instant::now(),
        })
    }

    /// Update the application
    fn update(&mut self) -> Result<()> {
        let t = (Instant::now() - self.start).as_secs_f32();
        let step = 2.0 * PI / 3.0;
        let a1 = step + t;
        let a2 = step * 2.0 + t;
        let a3 = step * 3.0 + t;

        self.particles.vertices = vec![
            particles::Vertex::new([a1.cos(), a1.sin()], [1.0, 0.0, 0.0, 1.0]),
            particles::Vertex::new([a2.cos(), a2.sin()], [0.0, 1.0, 0.0, 1.0]),
            particles::Vertex::new([a3.cos(), a3.sin()], [0.0, 0.0, 1.0, 1.0]),
        ];

        // no-op
        Ok(())
    }

    /// Draw the screen.
    fn render(&mut self) -> Result<()> {
        let particle_draw_commands = self.particles.draw(&self.display)?;
        self.display.render(vec![particle_draw_commands])?;
        Ok(())
    }

    /// Rebuild the swapchain and command buffers
    fn rebuild_swapchain_resources(&mut self) -> Result<()> {
        self.display.rebuild_swapchain()?;
        self.particles.rebuild_swapchain_resources(&self.display)?;
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

                Event::MainEventsCleared => {
                    match self.update().and_then(|_| self.render()) {
                        Err(error) => {
                            log::error!("unable to render the frame {}", error);
                            *control_flow = ControlFlow::Exit;
                        }
                        Ok(_) => {
                            self.display.surface.window().request_redraw();
                        }
                    }
                }

                _ => (),
            }
        });
    }
}
