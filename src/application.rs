mod particles;

use crate::display::Display;
use anyhow::{Context, Result};
use particles::Particles;
use std::time::Instant;
use winit::{
    event::{
        ElementState, Event, KeyboardInput, MouseButton, VirtualKeyCode,
        WindowEvent,
    },
    event_loop::ControlFlow,
    window::Fullscreen,
};

type Vec2 = nalgebra::Vector2<f32>;

pub struct Application {
    display: Display,
    particles: Particles,
    last_update: Instant,
    screen_dims: Vec2,
    mouse: Vec2,
    pressed: bool,
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
            screen_dims: [1.0, 1.0].into(),
            mouse: [0.0, 0.0].into(),
            pressed: false,
        })
    }

    /// Tick the application state based on the wall-clock time since the
    /// last tick.
    fn tick(&mut self, time: f32) -> Result<()> {
        let constants = particles::PushConstants {
            enabled: if self.pressed { 1 } else { 0 },
            attractor: self.mouse.into(),
            timestep: time,
            ..Default::default()
        };
        self.particles.tick(&self.display, constants)
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

        let [width, height] = self.display.swapchain.dimensions();
        self.screen_dims.x = width as f32;
        self.screen_dims.y = height as f32;

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
        self.display
            .surface
            .window()
            .set_fullscreen(Some(Fullscreen::Borderless(None)));

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
                    event: WindowEvent::CursorMoved { position, .. },
                    ..
                } => {
                    let world_width = self.screen_dims.x / self.screen_dims.y;
                    self.mouse.y =
                        lerp(position.y as f32 / self.screen_dims.y, 1.0, -1.0);
                    self.mouse.x = lerp(
                        position.x as f32 / self.screen_dims.x,
                        -world_width,
                        world_width,
                    );
                }

                Event::WindowEvent {
                    event:
                        WindowEvent::KeyboardInput {
                            input:
                                KeyboardInput {
                                    state: ElementState::Released,
                                    virtual_keycode: Some(VirtualKeyCode::Space),
                                    ..
                                },
                            ..
                        },
                    ..
                } => match self.particles.reset_vertices(&self.display) {
                    Err(error) => {
                        log::error!(
                            "problem while reseting vertices {:?}",
                            error
                        );
                        *control_flow = ControlFlow::Exit;
                    }
                    _ => {}
                },

                Event::WindowEvent {
                    event:
                        WindowEvent::MouseInput {
                            button: MouseButton::Left,
                            state,
                            ..
                        },
                    ..
                } => {
                    self.pressed = match state {
                        ElementState::Pressed => true,
                        ElementState::Released => false,
                    }
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

                _ => {}
            }
        });
    }
}

fn lerp(x: f32, min: f32, max: f32) -> f32 {
    x * max + (1.0 - x) * min
}
