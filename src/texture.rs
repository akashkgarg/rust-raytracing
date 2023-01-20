use crate::types::*;
use crate::utils::*;
use js_sys::Math::sin;
use rand::rngs::ThreadRng;
use rand::Rng;

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

const POINT_COUNT: usize = 256;

#[derive(Clone)]
pub struct PerlinTexture {
    perm_x: [usize; POINT_COUNT],
    perm_y: [usize; POINT_COUNT],
    perm_z: [usize; POINT_COUNT],
    random: [f32; POINT_COUNT],
    ranvec: [Vec3; POINT_COUNT],
    scale: f32
}

impl PerlinTexture {
    pub fn new(rng: &mut ThreadRng, scale: f32) -> Self {
        let mut random = [0_f32; POINT_COUNT];
        let mut ranvec = [Vec3::new(0.0,0.0,0.0); POINT_COUNT];
        for i in 0..POINT_COUNT {
            random[i] = rng.gen::<f32>();
            ranvec[i] = random_in_unit_sphere(rng).into_inner();
        }
        let perm_x = Self::permute(rng);
        let perm_y = Self::permute(rng);
        let perm_z = Self::permute(rng);

        PerlinTexture{random: random,
                      ranvec: ranvec,
                      perm_x: perm_x,
                      perm_y: perm_y,
                      perm_z: perm_z,
                      scale: scale}
    }

    fn permute(rng: &mut ThreadRng) -> [usize; POINT_COUNT] {
        let mut indices = [0; POINT_COUNT];
        for i in 0..POINT_COUNT {
            indices[i] = i;
        }

        // shuffle
        for i in (1..POINT_COUNT).rev() {
            let target = rng.gen_range(0..i+1);
            let tmp = indices[i];
            indices[i] = indices[target];
            indices[target] = tmp;
        }

        indices
    }

    fn trilinear_interp(c: [[[f32; 2]; 2]; 2], u: f32, v: f32, w: f32) -> f32 {
        let mut accum = 0.0;
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    accum += (i as f32 * u + (1-i) as f32 * (1.0-u) as f32) *
                        (j as f32 * v + (1-j) as f32 * (1.0-v) as f32) *
                        (k as f32 * w + (1-k) as f32 * (1.0-w) as f32) *
                        c[i][j][k];
                }
            }
        }
        return accum;
    }

    fn perlin_interp(c: [[[Vec3; 2]; 2]; 2], u: f32, v: f32, w: f32) -> f32 {
        // hermitian smoothing
        let uu = u*u*(3.0-2.0*u);
        let vv = v*v*(3.0-2.0*v);
        let ww = w*w*(3.0-2.0*w);

        let mut accum = 0.0;
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    let weight_v = Vec3::new(u-i as f32, v-j as f32, w-k as f32);
                    accum += (i as f32 * uu + (1-i) as f32 * (1.0-uu) as f32) *
                        (j as f32 * vv + (1-j) as f32 * (1.0-vv) as f32) *
                        (k as f32 * ww + (1-k) as f32 * (1.0-ww) as f32) *
                        (c[i][j][k].dot(&weight_v));
                }
            }
        }
        return accum;
    }

    // repeated calls to noise to layer noise functions together creating
    // turbulence.
    fn turbulence(&self, p: &Point, depth: usize) -> f32 {
        let mut accum = 0.0;
        let mut tmp_p = *p;
        let mut weight = 1.0;

        for _ in 0..depth {
            accum += weight*self.noise(&tmp_p);
            weight *= 0.5;
            tmp_p *= 2.0;
        }

        accum.abs()
    }

    fn noise(&self, p: &Point) -> f32 {

        let mut u = p.x - p.x.floor();
        let mut v = p.y - p.y.floor();
        let mut w = p.z - p.z.floor();

        let i = p.x.floor() as isize;
        let j = p.y.floor() as isize;
        let k = p.z.floor() as isize;

        // tri linear interp
        //let mut c = [[[0_f32; 2]; 2]; 2];
        let mut c = [[[Vec3::new(0.0,0.0,0.0); 2]; 2]; 2];
        for di in 0..2_isize {
            for dj in 0..2_isize {
                for dk in 0..2_isize {
                    c[di as usize][dj as usize][dk as usize] =
                        self.ranvec[self.perm_x[((i + di) & (POINT_COUNT-1) as isize) as usize] ^
                                    self.perm_y[((j + dj) & (POINT_COUNT-1) as isize) as usize] ^
                                    self.perm_z[((k + dk) & (POINT_COUNT-1) as isize) as usize]]
                }
            }
        }

        //Self::trilinear_interp(c, u, v, w)
        Self::perlin_interp(c, u, v, w)

        // this would be all you need for regular perlin without smoothing.
        // negative values will overflow when converting to usize, so we
        // take absolute value here.
        // let i = ((4.0*p.x).floor().abs() as usize) & POINT_COUNT-1;
        // let j = ((4.0*p.y).floor().abs() as usize) & POINT_COUNT-1;
        // let k = ((4.0*p.z).floor().abs() as usize) & POINT_COUNT-1;
        // return self.random[self.perm_x[i] ^ self.perm_y[j] ^ self.perm_z[k]];
    }
}

impl Texture for PerlinTexture {
    fn value(&self, _rng: &mut ThreadRng, _u: f32, _v: f32, p: &Point) -> Vec3 {
        // need to shift by 1.0 and multiply by 0.5 because
        // the output of the perlin interpretation can return negative values.
        // These negative values will be passed to the sqrt() function of our
        // gamma function and get turned into NaNs. We will cast the perlin
        // output back to between 0 and 1.
        //Vec3::new(1.0, 1.0, 1.0) * 0.5 * (1.0 + self.noise(&(self.scale * p)))

        // turbulence
        // Vec3::new(1.0, 1.0, 1.0) * self.turbulence(&(self.scale * p), 7)

        // marble effect by adjusting the phase
        Vec3::new(1.0, 1.0, 1.0) * 0.5 * (1.0 + sin((self.scale * p.z + 10.0 * self.turbulence(&p, 7)).into()) as f32)
    }
}
