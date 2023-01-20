
use rand::Rng;
use rand::rngs::ThreadRng;

use crate::types::*;

pub fn set_panic_hook() {
    // When the `console_error_panic_hook` feature is enabled, we can call the
    // `set_panic_hook` function at least once during initialization, and then
    // we will get better error messages if our code ever panics.
    //
    // For more details see
    // https://github.com/rustwasm/console_error_panic_hook#readme
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}


// A macro to provide `println!(..)`-style syntax for `console.log` logging.
macro_rules! log {
    ( $( $t:tt )* ) => {
        web_sys::console::log_1(&format!( $( $t )* ).into());
    }
}

pub(crate) use log;

pub(crate) fn random_in_unit_sphere(rng: &mut ThreadRng) -> UnitVec3 {
    let p = Vec3::new(rng.gen_range(-1.0..1.0),
                      rng.gen_range(-1.0..1.0),
                      rng.gen_range(-1.0..1.0));
    return UnitVec3::new_normalize(p);
}

pub(crate) fn random_in_hemisphere(rng: &mut ThreadRng, normal: &Vec3) -> UnitVec3 {
    let in_unit_sphere = random_in_unit_sphere(rng);
    if in_unit_sphere.dot(&normal) > 0.0 {
        // In the same hemisphere as the normal
        return in_unit_sphere;
    }
    else {
        return -in_unit_sphere;
    }
}

pub(crate) fn random_in_unit_disk(rng: &mut ThreadRng) -> Vec3 {
    loop {
        let p = Vec3::new(rng.gen_range(-1.0..1.0),
                          rng.gen_range(-1.0..1.0),
                          0.0);
        if p.dot(&p) >= 1.0 {
            continue;
        }
        return p;
    }
}
