use crate::application::particles::Vertex;

type Vec2 = nalgebra::Vector2<f32>;

const MAX_VEL: f32 = 5.0;
const DAMPING: f32 = 1.0 - 1e-5;

/// A particle which is simulated via kinematic equations.
#[derive(Debug, Copy, Clone)]
pub struct Particle {
    pub pos: Vec2,
    pub vel: Vec2,
    pub acc: Vec2,
    pub lifetime: f32,
}

impl Particle {
    pub fn at(pos: Vec2) -> Self {
        Self {
            pos,
            vel: Vec2::new(0.0, 0.0),
            acc: Vec2::new(0.0, 0.0),
            lifetime: 0.0,
        }
    }

    pub fn integrate(&mut self, time: f32) {
        self.vel += self.acc * time;
        self.clamp_vel();
        self.pos += self.vel * time;
        self.clamp_pos();
        self.lifetime += time;
        self.acc = Vec2::new(0.0, 0.0);
    }

    fn clamp_pos(&mut self) {
        self.pos.x = nalgebra::clamp(self.pos.x, -2.0, 2.0);
        self.pos.y = nalgebra::clamp(self.pos.y, -2.0, 2.0);
    }

    fn clamp_vel(&mut self) {
        if self.vel.magnitude_squared() > MAX_VEL * MAX_VEL {
            self.vel.normalize_mut();
            self.vel.scale_mut(MAX_VEL);
        }
        self.vel *= DAMPING;
    }
}

impl From<Particle> for Vertex {
    fn from(particle: Particle) -> Self {
        let nvel = particle.vel.norm() / MAX_VEL;
        let r = lerp(nvel, 0.1, 0.6);
        let g = lerp(nvel, 0.1, 0.6);
        let b = lerp(nvel, 0.7, 1.0);
        Vertex::new(particle.pos.into(), [r, g, b, 1.0])
    }
}

fn lerp(x: f32, min: f32, max: f32) -> f32 {
    x * max + (1.0 - x) * min
}
