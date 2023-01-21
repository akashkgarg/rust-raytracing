mod utils;
mod types;
mod texture;

use types::*;
use texture::*;
use utils::*;

use std::f64;
use std::fmt;
use std::fmt::Debug;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen::Clamped;
use js_sys::Math::sqrt;
use js_sys::Math::pow;
use js_sys::Math::tan;
use js_sys::Math::acos;
use js_sys::Math::atan2;
use rand::Rng;
use rand::rngs::ThreadRng;
use almost;

#[derive(Copy,Clone,Debug)]
pub struct Ray {
    origin: Point,
    dir: UnitVec3
}

#[derive(Clone)]
pub struct HitRecord {
    pt: Point,
    n: UnitVec3,
    t: f32,
    u: f32,
    v: f32,
    front_face: bool,
    mat: Box<dyn Material>
}

pub trait Hittable {
    fn hit(&self, ray: &Ray, tmin: f32, tmax: f32) -> Option<HitRecord>;
}

impl Ray {
    pub fn at(&self, t: f32) -> Point {
        self.origin + self.dir.into_inner() * t
    }
}

// scatters an incoming ray and outputs the color attenuation and the scattered
// ray. Also optionally for emissive materials, e.g., lights, outputs a color
pub trait Material: MaterialClone {
    fn scatter(&self, rng: &mut ThreadRng, ray: &Ray, rec: &HitRecord) -> Option<(Vec3, Ray)>;
    fn emitted(&self, rng: &mut ThreadRng, u: f32, v: f32, p: &Point) -> Vec3 {
        Vec3::new(0.0, 0.0, 0.0)
    }
}

pub trait MaterialClone {
    fn clone_box(&self) -> Box<dyn Material>;
}

impl<T> MaterialClone for T
where
    T: 'static + Material + Clone
{
    fn clone_box(&self) -> Box<dyn Material> {
        Box::new(self.clone())
    }
}

// We can now implement Clone manually by forwarding to clone_box.
impl Clone for Box<dyn Material> {
    fn clone(&self) -> Box<dyn Material> {
        self.clone_box()
    }
}

#[derive(Clone)]
struct Lambertian {
    albedo: Box<dyn Texture>
}

#[derive(Clone)]
struct DiffuseLight {
    emit: Box<dyn Texture>
}

#[derive(Clone)]
struct Metal {
    albedo: Vec3,
    fuzz: f32
}

#[derive(Clone)]
struct Dielectric {
    ir: f32 // index of refraction
}

impl Dielectric {
    fn reflectance(cosine: f32, ref_idx: f32) -> f32 {
        // Use Schlick's approximation for reflectance.
        let mut r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
        r0 = r0*r0;
        return r0 + (1.0 - r0)*pow(1.0 - cosine as f64, 5.0) as f32;
    }
}

impl Material for Lambertian {
    fn scatter(&self, rng: &mut ThreadRng, _ray: &Ray, rec: &HitRecord) -> Option<(Vec3, Ray)> {
        let attenuation = self.albedo.value(rng, rec.u, rec.v, &rec.pt);
        let scatter_direction = rec.n.into_inner() + random_in_hemisphere(rng, &rec.n).into_inner();

        if scatter_direction.iter().any(|&v| almost::zero(v)) {
            return Some((attenuation, Ray{origin: rec.pt, dir: rec.n}));
        } else {
            let ray = Ray{origin: rec.pt,
                          dir: UnitVec3::new_normalize(scatter_direction)};
            return Some((attenuation, ray));
        }
    }
}

impl Material for DiffuseLight {
    fn scatter(&self, rng: &mut ThreadRng, _ray: &Ray, rec: &HitRecord) -> Option<(Vec3, Ray)> {
        None
    }
    fn emitted(&self, rng: &mut ThreadRng, u: f32, v: f32, p: &Point) -> Vec3 {
        self.emit.value(rng, u, v, p)
    }
}

impl Material for Metal {
    fn scatter(&self, rng: &mut ThreadRng, ray: &Ray, rec: &HitRecord) -> Option<(Vec3, Ray)> {
        let attenuation = self.albedo;
        let reflected = ray.dir.into_inner() - 2.0*ray.dir.dot(&rec.n)*rec.n.into_inner();
        let scattered = UnitVec3::new_normalize(reflected + self.fuzz * random_in_unit_sphere(rng).into_inner());
        if scattered.dot(&rec.n) > 0.0 {
            let scattered_ray = Ray{origin: rec.pt, dir: scattered};
            return Some((attenuation, scattered_ray));
        } else {
            return None
        }
    }
}

impl Material for Dielectric {
    fn scatter(&self, rng: &mut ThreadRng, ray: &Ray, rec: &HitRecord) -> Option<(Vec3, Ray)> {
        let attenuation = Vec3::new(1.0,1.0,1.0);
        let refraction_ratio = if rec.front_face { 1.0/self.ir } else { self.ir };

        // refract the ray
        let uv = ray.dir.into_inner();
        let n = rec.n.into_inner();
        let cos_theta = (-ray.dir.dot(&rec.n)).min(1.0);

        let sin_theta = sqrt((1.0 - cos_theta*cos_theta) as f64) as f32;

        let cannot_refract = refraction_ratio * sin_theta > 1.0;
        let direction;
        if cannot_refract || Dielectric::reflectance(cos_theta, refraction_ratio) > rng.gen::<f32>() {
            direction = uv - 2.0*uv.dot(&n)*n;
        } else {
            let r_out_perp = refraction_ratio * (uv + cos_theta * n);
            let r_out_parallel = -sqrt((1.0 - r_out_perp.dot(&r_out_perp)).abs() as f64) as f32 * n;
            direction = r_out_perp + r_out_parallel;
        }

        return Some((attenuation,
                     Ray{origin: rec.pt,
                         dir: UnitVec3::new_normalize(direction)}));
    }
}

struct Sphere {
    center: Point,
    radius: f32,
    mat: Box<dyn Material>
}

impl Sphere {
    fn get_sphere_uv(&self, p: &Point) -> (f32, f32) {
        // p: a given point on the sphere of radius one, centered at the origin.
        // u: returned value [0,1] of angle around the Y axis from X=-1.
        // v: returned value [0,1] of angle from Y=-1 to Y=+1.
        //     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
        //     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
        //     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>

        let pi = std::f32::consts::PI;
        let theta = acos(-p.y as f64) as f32;
        let phi = atan2(-p.z as f64, p.x as f64) as f32 + pi;

        let u = phi / (2.0*pi);
        let v = theta / pi;

        (u, v)
    }
}

impl Hittable for Sphere {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let oc = r.origin - self.center;
        let a = r.dir.dot(&r.dir);
        let half_b = oc.dot(&r.dir);
        let c = oc.dot(&oc) - self.radius*self.radius;

        let discriminant = half_b*half_b - a*c;
        if discriminant < 0.0 {
            return None
        }

        let sqrtd = sqrt(discriminant as f64) as f32;

        // Find the nearest root that lies in the acceptable range.
        let mut root = (-half_b - sqrtd) / a;
        if root < t_min || t_max < root {
            root = (-half_b + sqrtd) / a;
            if root < t_min || t_max < root {
                return None
            }
        }

        // we divide by radius here even though we are normalizing since a
        // negative radius will invert the normals, which might be something we
        // want to glass spheres.
        let n = UnitVec3::new_normalize((r.at(root) - self.center) / self.radius);
        let front_face = r.dir.dot(&n) <= 0_f32;

        let (u,v) = self.get_sphere_uv(&r.at(root));

        return Some(HitRecord{pt: r.at(root),
                              n: if front_face { n } else { -n },
                              t: root,
                              u: u,
                              v: v,
                              front_face: front_face,
                              mat: self.mat.clone()})
    }
}

struct XYRect {
    x0: f32,
    x1: f32,
    y0: f32,
    y1: f32,
    k: f32,
    mat: Box<dyn Material>
}

impl XYRect {
    fn new(x0: f32, x1: f32, y0: f32, y1: f32, k: f32, mat: Box<dyn Material>) -> Self {
        Self{x0: x0, x1: x1, y0: y0, y1: y1, k: k, mat: mat}
    }
}

impl Hittable for XYRect {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let t = (self.k - r.origin.z) / r.dir.z;
        if t < t_min || t > t_max {
            return None
        }
        let x = r.origin.x + t*r.dir.x;
        let y = r.origin.y + t*r.dir.y;
        if x < self.x0 || x > self.x1 || y < self.y0 || y > self.y1 {
            return None
        }

        let u = (x-self.x0)/(self.x1-self.x0);
        let v = (y-self.y0)/(self.y1-self.y0);
        let n = Vec3::z_axis();
        let front_face = r.dir.dot(&n) <= 0_f32;

        return Some(HitRecord{pt: r.at(t),
                              n: if front_face { n } else { -n },
                              t: t,
                              u: u,
                              v: v,
                              front_face: front_face,
                              mat: self.mat.clone()})
    }
}

struct XZRect {
    x0: f32,
    x1: f32,
    z0: f32,
    z1: f32,
    k: f32,
    mat: Box<dyn Material>
}

impl XZRect {
    fn new(x0: f32, x1: f32, z0: f32, z1: f32, k: f32, mat: Box<dyn Material>) -> Self {
        Self{x0: x0, x1: x1, z0: z0, z1: z1, k: k, mat: mat}
    }
}

impl Hittable for XZRect {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let t = (self.k - r.origin.y) / r.dir.y;
        if t < t_min || t > t_max {
            return None
        }
        let x = r.origin.x + t*r.dir.x;
        let z = r.origin.z + t*r.dir.z;
        if x < self.x0 || x > self.x1 || z < self.z0 || z > self.z1 {
            return None
        }

        let u = (x-self.x0)/(self.x1-self.x0);
        let v = (z-self.z0)/(self.z1-self.z0);
        let n = Vec3::y_axis();
        let front_face = r.dir.dot(&n) <= 0_f32;

        return Some(HitRecord{pt: r.at(t),
                              n: if front_face { n } else { -n },
                              t: t,
                              u: u,
                              v: v,
                              front_face: front_face,
                              mat: self.mat.clone()})
    }
}

struct YZRect {
    y0: f32,
    y1: f32,
    z0: f32,
    z1: f32,
    k: f32,
    mat: Box<dyn Material>
}

impl YZRect {
    fn new(y0: f32, y1: f32, z0: f32, z1: f32, k: f32, mat: Box<dyn Material>) -> Self {
        Self{y0: y0, y1: y1, z0: z0, z1: z1, k: k, mat: mat}
    }
}

impl Hittable for YZRect {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let t = (self.k - r.origin.x) / r.dir.x;
        if t < t_min || t > t_max {
            return None
        }
        let y = r.origin.y + t*r.dir.y;
        let z = r.origin.z + t*r.dir.z;
        if y < self.y0 || y > self.y1 || z < self.z0 || z > self.z1 {
            return None
        }

        let u = (y-self.y0)/(self.y1-self.y0);
        let v = (z-self.z0)/(self.z1-self.z0);
        let n = Vec3::x_axis();
        let front_face = r.dir.dot(&n) <= 0_f32;

        return Some(HitRecord{pt: r.at(t),
                              n: if front_face { n } else { -n },
                              t: t,
                              u: u,
                              v: v,
                              front_face: front_face,
                              mat: self.mat.clone()})
    }
}

#[wasm_bindgen]
#[derive(Copy, Clone, Debug)]
pub struct Camera {
    origin: Point,
    lower_left_corner: Point,
    horizontal: Vec3,
    vertical: Vec3,
    u: Vec3,
    v: Vec3,
    w: Vec3,
    lens_radius: f32
}

#[wasm_bindgen]
impl Camera {
    pub fn new(lookfrom: &js_sys::Array,
               lookat: &js_sys::Array,
               vup: &js_sys::Array,
               vfov: f32,
               aspect: f32,
               aperture: f32) -> Self {
        let theta = vfov * (std::f32::consts::PI / 180.0);
        let h = tan(theta as f64/ 2.0 ) as f32;

        log!("Got array: {:?}", lookfrom);

        let viewport_height: f32 = 2.0 * h;
        let viewport_width: f32 = aspect * viewport_height;
        let focal_length: f32 = 1.0;

        let from: Point = Point::new(lookfrom.at(0).as_f64().unwrap() as f32,
                                     lookfrom.at(1).as_f64().unwrap() as f32,
                                     lookfrom.at(2).as_f64().unwrap() as f32);
        let to: Point = Point::new(lookat.at(0).as_f64().unwrap() as f32,
                                  lookat.at(1).as_f64().unwrap() as f32,
                                  lookat.at(2).as_f64().unwrap() as f32);
        let up = Vec3::new(vup.at(0).as_f64().unwrap() as f32,
                           vup.at(1).as_f64().unwrap() as f32,
                           vup.at(2).as_f64().unwrap() as f32);

        let w = UnitVec3::new_normalize(from - to);
        let u = UnitVec3::new_normalize(up.cross(&w));
        let v = w.cross(&u);

        let focus_dist: f32 = (to - from).norm();

        let origin: Point = from;
        let horizontal: Vec3 = focus_dist * viewport_width * u.into_inner();
        let vertical: Vec3 = focus_dist * viewport_height * v;
        let lower_left_corner: Point = origin - horizontal*0.5_f32 - vertical*0.5_f32 - focus_dist*w.into_inner();

        Camera{origin: origin,
               lower_left_corner: lower_left_corner,
               horizontal: horizontal,
               vertical: vertical,
               u: u.into_inner(),
               v: v,
               w: w.into_inner(),
               lens_radius: aperture / 2.0}
    }
}

impl Camera {
    pub fn get_ray(&self, rng: &mut ThreadRng, s: f32, t: f32) -> Ray {
        let rd = self.lens_radius * random_in_unit_disk(rng);
        let offset = self.u * rd.x + self.v * rd.y;
        Ray{origin: self.origin + offset,
            dir: UnitVec3::new_normalize(self.lower_left_corner +
                                            s*self.horizontal + t*self.vertical - self.origin - offset)}
    }
}

fn hit_sphere(center: &Point, radius: f32, r: &Ray) -> f32 {
    let oc = r.origin - center;
    let a = r.dir.dot(&r.dir);
    let half_b = oc.dot(&r.dir);
    let c = oc.dot(&oc) - radius*radius;
    let discriminant = half_b*half_b - a*c;
    if discriminant < 0.0 {
        return -1.0;
    } else {
        return (-half_b - sqrt(discriminant as f64) as f32 ) / a;
    }
}

fn ray_color(ray: &Ray, bg_color: &Vec3,
             world: &Vec<Box<dyn Hittable>>, rng: &mut ThreadRng, depth: u32) -> Vec3 {

    // exceeded ray bounce limit, no more light is gathered
    if depth <= 0 {
        return Vec3::new(0.0,0.0,0.0);
    }

    // get the closest hit pt from the ray to world
    let closest_hit = world.iter().map(|item| item.hit(&ray, 0.001, f32::MAX))
                                  .filter(|val| val.is_some())
                                  .min_by(|a,b| {
                                      let at: f32 = a.as_ref().unwrap().t;
                                      let bt: f32 = b.as_ref().unwrap().t;
                                      at.total_cmp(&bt)
                                  });
    match closest_hit {
        // if there is a closest hit
        Some(hit) => {
            match hit {
                // get the hit record if there is one
                Some(rec) =>  {
                    let emitted = rec.mat.emitted(rng, rec.u, rec.v, &rec.pt);
                    let color = match rec.mat.scatter(rng, &ray, &rec) {
                        Some((attenuation, scattered_ray)) => {
                            emitted + attenuation.component_mul(&ray_color(&scattered_ray, bg_color, world, rng, depth - 1))
                        },
                        None => emitted
                    };
                    return color;
                },
                _ => { return Vec3::new(0.0,0.0,0.0) }
            }
        },
        _ => {}
    }

    // for item in world {
    //     match item.hit(&ray, 0.001, f32::MAX) {
    //         None => continue,
    //         Some(rec) => {
    //             if rec.t > 0.0 {
    //                 let color = match rec.mat.scatter(rng, &ray, &rec) {
    //                     Some((attenuation, scattered_ray)) => {
    //                         attenuation.component_mul(&ray_color(&scattered_ray, world, rng, depth - 1))
    //                     },
    //                     None => Vec3::new(0.0,0.0,0.0)
    //                 };
    //                 return color;
    //             }
    //         }
    //     }
    // }

    // background
    *bg_color
    // let t = 0.5*(ray.dir.y + 1.0);
    // Vec3::new(1.0-t, 1.0-t, 1.0-t) + Vec3::new(0.5 * t, 0.7 * t, t)
}

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen(start)]
pub fn start() {
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document.get_element_by_id("raytracing-canvas").unwrap();
    let canvas: web_sys::HtmlCanvasElement = canvas
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .map_err(|_| ())
        .unwrap();

    let context = canvas
        .get_context("2d")
        .unwrap()
        .unwrap()
        .dyn_into::<web_sys::CanvasRenderingContext2d>()
        .unwrap();

    context.begin_path();

    // Draw the outer circle.
    context
        .arc(75.0, 75.0, 50.0, 0.0, f64::consts::PI * 2.0)
        .unwrap();

    // Draw the mouth.
    context.move_to(110.0, 75.0);
    context.arc(75.0, 75.0, 35.0, 0.0, f64::consts::PI).unwrap();

    // Draw the left eye.
    context.move_to(65.0, 65.0);
    context
        .arc(60.0, 65.0, 5.0, 0.0, f64::consts::PI * 2.0)
        .unwrap();

    // Draw the right eye.
    context.move_to(95.0, 65.0);
    context
        .arc(90.0, 65.0, 5.0, 0.0, f64::consts::PI * 2.0)
        .unwrap();

    context.stroke();
    log!("called start!");
}

#[wasm_bindgen]
pub fn draw(
    ctx: &web_sys::CanvasRenderingContext2d,
    camera: &Camera,
    width: u32,
    height: u32
) -> Result<(), JsValue> {
    let mut data = render(camera, width, height);
    let data = web_sys::ImageData::new_with_u8_clamped_array_and_sh(Clamped(&mut data), width, height)?;
    ctx.put_image_data(&data, 0.0, 0.0)
}

fn sphere_diffuse_light(rng: &mut ThreadRng) -> Vec<Box<dyn Hittable>> {
    let mut world: Vec<Box<dyn Hittable>> = Vec::new();
    let perlin1 = Box::new(Lambertian{albedo: Box::new(PerlinTexture::new(rng, 4.0))});
    let perlin2 = Box::new(Lambertian{albedo: Box::new(PerlinTexture::new(rng, 4.0))});
    world.push(Box::new(Sphere{center: Point::new(0.0, -1000.0, 0.0), radius: 1000.0, mat: perlin1}));
    world.push(Box::new(Sphere{center: Point::new(0.0, 2.0, 0.0), radius: 2.0, mat: perlin2}));

    let difflight = Box::new(DiffuseLight{emit: Box::new(ConstantTexture{color: Vec3::new(4.0, 4.0, 4.0)})});
    world.push(Box::new(Sphere{center: Point::new(0.0, 8.0, 0.0), radius: 2.0, mat: difflight.clone()}));
    world.push(Box::new(XYRect::new(3.0, 5.0, 1.0, 3.0, -2.0, difflight)));
    world
}

fn empty_cornell_box(rng: &mut ThreadRng) -> Vec<Box<dyn Hittable>> {
    let mut world: Vec<Box<dyn Hittable>> = Vec::new();
    let red = Box::new(Lambertian{albedo: Box::new(ConstantTexture{color: Vec3::new(0.65, 0.05, 0.05)})});
    let white = Box::new(Lambertian{albedo: Box::new(ConstantTexture{color: Vec3::new(0.73, 0.73, 0.73)})});
    let green = Box::new(Lambertian{albedo: Box::new(ConstantTexture{color: Vec3::new(0.12, 0.45, 0.15)})});
    let light = Box::new(DiffuseLight{emit: Box::new(ConstantTexture{color: Vec3::new(15.0, 15.0, 15.0)})});

    world.push(Box::new(YZRect::new(0.0, 555.0, 0.0, 555.0, 555.0, green.clone())));
    world.push(Box::new(YZRect::new(0.0, 555.0, 0.0, 555.0, 0.0, red.clone())));

    world.push(Box::new(XZRect::new(213.0, 343.0, 227.0, 332.0, 554.0, light.clone())));

    world.push(Box::new(XZRect::new(0.0, 555.0, 0.0, 555.0, 0.0, white.clone())));
    world.push(Box::new(XZRect::new(0.0, 555.0, 0.0, 555.0, 555.0, white.clone())));
    world.push(Box::new(XYRect::new(0.0, 555.0, 0.0, 555.0, 555.0, white.clone())));

    world
}

fn render(camera: &Camera, width: u32, height: u32) -> Vec<u8>{

    let mut rng = rand::thread_rng();

    let samples_per_pixel = 10;
    let max_depth = 10;

    let bg_color = Vec3::new(0.0, 0.0, 0.0);

    // world
    //let world = sphere_diffuse_light(&mut rng);
    let world = empty_cornell_box(&mut rng);

    // materials
    //let material_ground = Box::new(Lambertian{albedo: Box::new(ConstantTexture{color: Vec3::new(0.8, 0.8, 0.0)})});
    //let material_ground = Box::new(Lambertian{albedo: Box::new(CheckerTexture::new(Vec3::new(0.3, 0.2, 0.1),
    //let material_center = Box::new(Lambertian{albedo: Vec3::new(0.7, 0.3, 0.3)});
    //let material_center = Box::new(Lambertian{albedo: Box::new(ConstantTexture{color: Vec3::new(0.1, 0.2, 0.5)})});
    // let material_left = Box::new(Metal{albedo: Vec3::new(0.8, 0.8, 0.8), fuzz: 0.3});
    //let material_left = Box::new(Dielectric{ir: 1.5});
    //let material_right = Box::new(Metal{albedo: Vec3::new(0.8, 0.6, 0.2), fuzz: 0.0});

    // world.push(Sphere{center: Point::new(0.0, -100.5, -1.0), radius: 100.0, mat: material_ground});
    // world.push(Sphere{center: Point::new(0.0, 0.0, -1.0), radius: 0.5, mat: material_center});
    // world.push(Sphere{center: Point::new(-1.0, 0.0, -1.0), radius: 0.5, mat: material_left.clone()});
    // world.push(Sphere{center: Point::new(-1.0, 0.0, -1.0), radius: -0.4, mat: material_left.clone()});
    // world.push(Sphere{center: Point::new(1.0, 0.0, -1.0), radius: 0.5, mat: material_right});

    // render

    let mut data = Vec::new();
    // image data goes from top to bottom
    for j in (0..height).rev() {
        for i in 0..width {

            let mut pixel_color: Vec3 = Vec3::new(0.0,0.0,0.0);

            for _ in 0..samples_per_pixel {
                let u = (i as f32 + rng.gen::<f32>()) / (width-1) as f32;
                let v = (j as f32 + rng.gen::<f32>()) / (height-1) as f32;
                // let u = (i as f32) / (width-1) as f32;
                // let v = (j as f32) / (height-1) as f32;
                let ray = camera.get_ray(&mut rng, u, v);
                pixel_color += ray_color(&ray, &bg_color, &world, &mut rng, max_depth);
            }

            pixel_color /= samples_per_pixel as f32;

            // gamma correct for gamma = 2
            pixel_color.x = sqrt(pixel_color.x as f64) as f32;
            pixel_color.y = sqrt(pixel_color.y as f64) as f32;
            pixel_color.z = sqrt(pixel_color.z as f64) as f32;

            let pixel = Color::new((pixel_color.x * 255_f32).round() as u8,
                                   (pixel_color.y * 255_f32).round() as u8,
                                   (pixel_color.z * 255_f32).round() as u8,
                                   255);

            // let r = i as f64 / (width-1) as f64;
            // let g = j as f64 / (height-1) as f64;
            // let b = 0.25;

            // let ir: u8 = (255.999_f64 * r).round() as u8;
            // let ig: u8 = (255.999_f64 * g).round() as u8;
            // let ib: u8 = (255.999_f64 * b).round() as u8;
            data.push(pixel.x);
            data.push(pixel.y);
            data.push(pixel.z);
            data.push(pixel.w);
        }
    }

    data
}

#[wasm_bindgen]
extern {
    fn alert(s: &str);
}

#[wasm_bindgen]
pub fn greet() {
    alert("Hello, raytracing!");
}
