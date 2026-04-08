use std::path::Path;
use std::{error::Error, fs::File};

use chrono::{self, Local};
use rand::prelude::*;

use super::{
    Float, PI, Vec3, World,
    math::{Bounded, UnitVec},
};

const MEGA: Float = 1e6;
const KILO: Float = 1e3;
const NANO: Float = 1e-9;
#[derive(Clone, Copy, serde::Serialize)]
pub struct WorldBuilder {
    pub fill_fraction: Float,
    pub particle_number: usize,
    pub small_saxis: Float,
    pub big_saxis: Float,

    pub mag_moment_density: Float,

    pub viscosity: Float,

    pub epsilon_mat: Float,
    pub epsilon_part: Float,

    pub h_field: Vec3,
    pub e_field: Vec3,

    pub repulsion_factor: Float,

    pub delta_time: Float,
    pub duration: Float,

    pub log_step: usize,
    pub seed: Option<u64>,
}

impl Default for WorldBuilder {
    fn default() -> Self {
        let small_saxis: Float = 75.0 * NANO;
        let mag_moment_density: Float = 380.0 * KILO;

        let h_field: Vec3 = Vec3::new(0.0, 0.0, 1.0) * 500.0 * mag_moment_density;
        let e_field: Vec3 = Vec3::new(0.0, 1.0, 0.0) * 100.0 * MEGA;
        Self {
            fill_fraction: 0.01,
            particle_number: 20,
            small_saxis: 75.0 * NANO,
            big_saxis: small_saxis * 3.5,
            mag_moment_density: 380.0 * KILO,
            viscosity: 3.5,
            epsilon_mat: 2.0,
            epsilon_part: 10.0,
            h_field,
            e_field,
            repulsion_factor: 10000.0,
            delta_time: 0.0001,
            duration: 20.0,
            log_step: 1,
            seed: None,
        }
    }
}

impl WorldBuilder {
    pub fn build(mut self) -> World {
        let particle_vol = 4.0 / 3.0 * PI * self.big_saxis.powi(2) * self.small_saxis;
        let rve_side_len =
            (particle_vol * self.particle_number as Float / self.fill_fraction).powf(1.0 / 3.0);
        let radius_eq = (self.big_saxis.powi(2) * self.small_saxis).powf(1.0 / 3.0);
        let t_drag = 6.0 * PI * self.viscosity * radius_eq;
        let r_drag = 8.0 * PI * self.viscosity * self.big_saxis.powi(2) * self.small_saxis;
        let shape_factor = self.small_saxis * self.big_saxis / 2.0
            * (PI / 2.0 + self.small_saxis / self.big_saxis);
        let mag_sus_x = (self.epsilon_part - self.epsilon_mat)
            / (1.0 + (self.epsilon_part - self.epsilon_mat) / self.epsilon_mat * shape_factor);
        let mag_sus_y = (self.epsilon_part - self.epsilon_mat)
            / (1.0 + (self.epsilon_part - self.epsilon_mat) / self.epsilon_mat * shape_factor);
        let mag_dipole = particle_vol * self.mag_moment_density;

        let log_dir = format!("out/{}", Local::now().format("%Y-%m-%d_%H-%M-%S"));
        if let Err(err) = std::fs::create_dir_all(&log_dir) {
            eprintln!("could not make log dir: {err}")
        }

        if let Err(err) = self.to_json(format!("{}/config.json", log_dir)) {
            eprintln!("could not log configuration: {err}")
        }

        let seed = self.seed.unwrap_or_else(rand::random::<u64>);
        self.seed = Some(seed);
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        World {
            positions: (&mut rng)
                .sample_iter(Bounded(rve_side_len))
                .take(self.particle_number)
                .collect(),
            directions: (&mut rng)
                .sample_iter(UnitVec)
                .take(self.particle_number)
                .collect(),
            el_dipole_moments: vec![Vec3::default(); self.particle_number],
            e_field: vec![Vec3::default(); self.particle_number],
            h_field: vec![Vec3::default(); self.particle_number],

            param: self,
            particle_vol,
            rve_side_len,
            radius_eq,
            t_drag,
            r_drag,
            e_sus_x: mag_sus_x,
            e_sus_z: mag_sus_y,
            mag_dipole,

            log_dir,
        }
    }

    pub fn to_json(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn Error>> {
        let file = File::create(path)?;
        serde_json::ser::to_writer_pretty(file, &self)?;
        Ok(())
    }
}
