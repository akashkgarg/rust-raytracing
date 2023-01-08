mod utils;

use std::f64;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen::Clamped;
use js_sys::Math::sqrt;
use nalgebra::{Point3, Vector3, Vector4, UnitVector3};
use rand::Rng;

// A macro to provide `println!(..)`-style syntax for `console.log` logging.
macro_rules! log {
    ( $( $t:tt )* ) => {
        web_sys::console::log_1(&format!( $( $t )* ).into());
    }
}

// RGBA color value, [0,255]
type Color = Vector4<u8>;
type Point = Point3<f32>;
type Vec3 = Vector3<f32>;
type UnitVec3 = UnitVector3<f32>;

#[derive(Copy,Clone,Debug)]
pub struct Ray {
    origin: Point,
    dir: UnitVec3
}

#[derive(Copy,Clone,Debug)]
pub struct HitRecord {
    pt: Point,
    n: UnitVec3,
    t: f32
}

pub trait Hittable {
    fn hit(&self, ray: &Ray, tmin: f32, tmax: f32) -> Option<HitRecord>;
}

impl Ray {
    pub fn at(&self, t: f32) -> Point {
        self.origin + self.dir.into_inner() * t
    }
}

struct Sphere {
    center: Point,
    radius: f32
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
        let root = (-half_b - sqrtd) / a;
        if root < t_min || t_max < root {
            let root2 = (-half_b + sqrtd) / a;
            if root2 < t_min || t_max < root2 {
                return None
            }
        }

        let n = UnitVector3::new_normalize(r.at(root) - self.center);
        let front_face = r.dir.dot(&n) < 0_f32;

        return Some(HitRecord{pt: r.at(root),
                              n: if front_face { n } else { -n },
                              t: root})
    }
}

#[derive(Copy, Clone, Debug)]
struct Camera {
    origin: Point,
    lower_left_corner: Point,
    horizontal: Vec3,
    vertical:Vec3
}

impl Camera {
    pub fn new() -> Self {
        let aspect: f32 = 16.0 / 9.0;

        let viewport_height: f32 = 2.0;
        let viewport_width: f32 = aspect * viewport_height;
        let focal_length: f32 = 1.0;

        let origin: Point = Point3::new(0_f32,0_f32,0_f32);
        let horizontal: Vec3 = Vector3::new(viewport_width, 0_f32, 0_f32);
        let vertical: Vec3 = Vector3::new(0_f32, viewport_height, 0_f32);
        let lower_left_corner: Point = origin - horizontal*0.5_f32 - vertical*0.5_f32 - Vector3::new(0_f32, 0_f32, focal_length);

        Camera{origin: origin,
               lower_left_corner: lower_left_corner,
               horizontal: horizontal,
               vertical: vertical}
    }

    pub fn get_ray(&self, u: f32, v: f32) -> Ray {
        Ray{origin: self.origin,
            dir: UnitVector3::new_normalize(self.lower_left_corner +
                                            u*self.horizontal + v*self.vertical - self.origin)}
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

fn ray_color<T: Hittable>(ray: &Ray, world: &Vec<T>) -> Color {
    for item in world {
        match item.hit(&ray, 0_f32, f32::MAX) {
            None => continue,
            Some(rec) => {
                if rec.t > 0.0 {
                    let color = 0.5 * Vector3::new(rec.n.x + 1.0,
                                                   rec.n.y + 1.0,
                                                   rec.n.z + 1.0);
                    return Color::new((255_f32 * color.x).round() as u8,
                                      (255_f32 * color.y).round() as u8,
                                      (255_f32 * color.z).round() as u8,
                                      255);
                }
            }
        }
    }
    // background
    let t = 0.5*(ray.dir.y + 1.0);
    Color::new((255_f32 * (1.0-t)).round() as u8,
               (255_f32 * (1.0-t)).round() as u8,
               (255_f32 * (1.0-t)).round() as u8,
               255) +
        Color::new((255_f32 * 0.5 * t).round() as u8,
                   (255_f32 * 0.7 * t).round() as u8,
                   (255_f32 * t).round() as u8,
                   255)
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
    width: u32,
    height: u32
) -> Result<(), JsValue> {
    let mut data = render(width, height);
    let data = web_sys::ImageData::new_with_u8_clamped_array_and_sh(Clamped(&mut data), width, height)?;
    ctx.put_image_data(&data, 0.0, 0.0)
}

fn render(width: u32, height: u32) -> Vec<u8>{

    let aspect: f32 = 16.0 / 9.0;
    let samples_per_pixel = 10;

    // world
    let mut world = Vec::new();
    world.push(Sphere{center: Point::new(0.0, 0.0, -1.0), radius: 0.5});
    world.push(Sphere{center: Point::new(0.0, -100.5, -1.0), radius: 100.0});

    // camera
    let camera = Camera::new();

    // render

    let mut data = Vec::new();
    // image data goes from top to bottom
    for j in (0..height).rev() {
        for i in 0..width {

            let mut pixel_color: Vec3 = Vec3::new(0.0,0.0,0.0);

            let mut rng = rand::thread_rng();

            for _ in 0..samples_per_pixel {
                let u = (i as f32 + rng.gen::<f32>()) / (width-1) as f32;
                let v = (j as f32 + rng.gen::<f32>()) / (height-1) as f32;
                let ray = camera.get_ray(u, v);
                let color = ray_color(&ray, &world);
                pixel_color += Vec3::new(color.x as f32 / 255_f32,
                                         color.y as f32 / 255_f32,
                                         color.z as f32 / 255_f32);
            }

            pixel_color /= samples_per_pixel as f32;

            // gamma correct

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
