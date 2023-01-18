use crate::types::*;

use js_sys::Math::sin;
use rand::rngs::ThreadRng;

// Calculate the color based on u,v and 3d point.
pub trait Texture : TextureClone {
    fn value(&self, rng: &mut ThreadRng, u: f32, v: f32, pt: &Point) -> Vec3;
}

pub trait TextureClone {
    fn clone_box(&self) -> Box<dyn Texture>;
}

impl<T> TextureClone for T
where
    T: 'static + Texture + Clone
{
    fn clone_box(&self) -> Box<dyn Texture> {
        Box::new(self.clone())
    }
}

// We can now implement Clone manually by forwarding to clone_box.
impl Clone for Box<dyn Texture> {
    fn clone(&self) -> Box<dyn Texture> {
        self.clone_box()
    }
}

//------------------------------------------------------------------------------

#[derive(Clone)]
pub struct ConstantTexture {
    pub(crate) color: Vec3
}

impl Texture for ConstantTexture {
    fn value(&self, _rng: &mut ThreadRng, _u: f32, _v: f32, _pt: &Point) -> Vec3 {
        self.color
    }
}

//------------------------------------------------------------------------------

#[derive(Clone)]
pub struct CheckerTexture {
    pub(crate) even: Box<dyn Texture>,
    pub(crate) odd: Box<dyn Texture>
}

impl CheckerTexture {
    pub fn new(c1: Vec3, c2: Vec3) -> Self {
        CheckerTexture{even: Box::new(ConstantTexture{color: c1}),
                       odd: Box::new(ConstantTexture{color: c2})}
    }
}

impl Texture for CheckerTexture {
    fn value(&self, rng: &mut ThreadRng, u: f32, v: f32, p: &Point) -> Vec3 {
        let sines = (sin(10.0*p.x as f64) * sin(10.0*p.y as f64)*sin(10.0*p.z as f64)) as f32;
        if sines < 0.0 {
            return self.odd.value(rng, u, v, p);
        } else {
            return self.even.value(rng, u, v, p);
        }
    }
}

//------------------------------------------------------------------------------
