mod kinematic_particle;
mod particles;

use crate::display::Display;
use anyhow::{Context, Result};
use kinematic_particle::Particle;
use particles::Particles;
use std::time::Instant;
use winit::{
    event::{ElementState, Event, MouseButton, WindowEvent},
    event_loop::ControlFlow,
};

type Vec2 = nalgebra::Vector2<f32>;

pub struct Application {
    display: Display,
    particles: Particles,
    last_update: Instant,
    kinematic: Vec<Particle>,
}

impl Application {
    pub fn initialize() -> Result<Self> {
        let display =
            Display::create().context("unable to create the display")?;
        let particles = Particles::new(&display)?;

        Ok(Self {
            display,
            particles,
            last_update: Instant::now(),
            kinematic: vec![Particle::at([0.0, 0.0].into())],
        })
    }

    fn spawn_particles(&mut self) {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let mut new_particles = (0..50)
            .into_iter()
            .map(|_| Particle::at([0.0, 0.0].into()))
            .map(|mut particle| {
                particle.vel.x = rng.gen_range(-1.0..1.0);
                particle.vel.y = rng.gen_range(-1.0..1.0);
                particle
            })
            .collect::<Vec<Particle>>();
        self.kinematic.append(&mut new_particles);
    }

    /// Tick the application state based on the wall-clock time since the
    /// last tick.
    fn tick(&mut self, _time: f32) -> Result<()> {
        self.particles.tick(&self.display)
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

    /// Update the application, only tick once every 15 milliseconds
    fn update(&mut self) -> Result<()> {
        const TICK_MILLIS: u128 = 15;
        let duration = Instant::now() - self.last_update;
        if duration.as_millis() >= TICK_MILLIS {
            self.tick(duration.as_secs_f32())?;
            self.last_update = Instant::now();
            Ok(())
        } else {
            Ok(())
        }
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
                    event:
                        WindowEvent::MouseInput {
                            button: MouseButton::Left,
                            state: ElementState::Released,
                            ..
                        },
                    ..
                } => {
                    self.spawn_particles();
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
