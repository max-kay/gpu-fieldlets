use std::path::Path;
use std::{error::Error, fs::File};

use chrono::{self, Local};
use rand::prelude::*;

use crate::math::GoodValues;

use super::{
    Float, PI, Simulation, Vec3,
    math::{Bounded, UnitVec},
};

const MEGA: Float = 1e6;
const KILO: Float = 1e3;
const NANO: Float = 1e-9;

#[derive(Clone, serde::Serialize)]
pub struct SimulationBuilder {
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
    pub name: String,
}

impl Default for SimulationBuilder {
    fn default() -> Self {
        let small_saxis: Float = 75.0 * NANO;
        let mag_moment_density: Float = 380.0 * KILO;

        let h_field: Vec3 = Vec3::new(0.0, 0.0, 1.0) * 5.0 * mag_moment_density;
        let e_field: Vec3 = Vec3::new(0.0, 0.0, 1.0) * 100.0 * MEGA;
        Self {
            fill_fraction: 0.01,
            particle_number: 1000,
            small_saxis: 75.0 * NANO,
            big_saxis: small_saxis * 3.5,
            mag_moment_density: 380.0 * KILO,
            viscosity: 3.5,
            epsilon_mat: 2.0,
            epsilon_part: 10.0,
            h_field,
            e_field,
            repulsion_factor: 40.0,
            delta_time: 0.00001,
            duration: 1.0,
            log_step: 1000,
            seed: None,
            name: String::new(),
        }
    }
}

impl SimulationBuilder {
    pub fn set_e_field_dir(mut self, v: Vec3) -> Self {
        let normalized = v.normalised();
        if normalized.is_finite() {
            self.e_field = 100.0 * MEGA * normalized;
        }
        self
    }

    pub fn set_h_field_dir(mut self, v: Vec3) -> Self {
        let normalized = v.normalised();
        if normalized.is_finite() {
            self.h_field = 5.0 * self.mag_moment_density * normalized;
        }
        self
    }

    pub fn h_field_set_zero(mut self) -> Self {
        self.h_field = Vec3::new(0.0, 0.0, 0.0);
        self
    }

    pub fn e_field_set_zero(mut self) -> Self {
        self.e_field = Vec3::new(0.0, 0.0, 0.0);
        self
    }

    pub fn build(mut self) -> Simulation {
        let particle_vol = 4.0 / 3.0 * PI * self.big_saxis.powi(2) * self.small_saxis;
        let rve_side_len =
            (particle_vol * self.particle_number as Float / self.fill_fraction).powf(1.0 / 3.0);
        let radius_eq = (self.big_saxis.powi(2) * self.small_saxis).powf(1.0 / 3.0);
        let t_drag = 6.0 * PI * self.viscosity * radius_eq;
        let r_drag = 8.0 * PI * self.viscosity * self.big_saxis.powi(2) * self.small_saxis;
        let shape_factor = self.small_saxis / (2.0 * self.big_saxis)
            * (PI / 2.0 + self.small_saxis / self.big_saxis);
        let e_sus_x = (self.epsilon_part - self.epsilon_mat)
            / (1.0 + (self.epsilon_part - self.epsilon_mat) / self.epsilon_mat * shape_factor);
        let e_sus_z = (self.epsilon_part - self.epsilon_mat)
            / (1.0
                + (self.epsilon_part - self.epsilon_mat) / self.epsilon_mat
                    * (1.0 - 2.0 * shape_factor));
        let mag_dipole = particle_vol * self.mag_moment_density;

        let seed = self.seed.unwrap_or_else(rand::random::<u64>);
        self.seed = Some(seed);
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut positions: Vec<Vec3> = Vec::with_capacity(self.particle_number);
        let min_dist = 2.0 * radius_eq * 1.1; // 10% clearance
        let mut attempts = 0;
        while positions.len() < self.particle_number {
            let candidate: Vec3 = rng.sample(Bounded(rve_side_len));
            let overlap = positions.iter().any(|p| {
                let r = (candidate - *p) % rve_side_len;
                r.norm() < min_dist
            });
            if !overlap {
                positions.push(candidate);
            }
            attempts += 1;
            if attempts > self.particle_number * 10_000 {
                eprintln!(
                    "Warning: could not place all particles without overlap after {attempts} attempts"
                );
                break;
            }
        }

        let log_dir = format!("out/{}", Local::now().format("%Y-%m-%d_%H-%M-%S"));
        if let Err(err) = std::fs::create_dir_all(&log_dir) {
            eprintln!("could not make log dir: {err}")
        }

        if let Err(err) = self.to_json(format!("{}/config.json", log_dir)) {
            eprintln!("could not log configuration: {err}")
        }

        let mut this = Simulation {
            positions,
            directions: (&mut rng)
                .sample_iter(UnitVec)
                .take(self.particle_number)
                .collect(),
            el_dipole_moments: vec![Vec3::default(); self.particle_number],
            e_field: vec![self.e_field; self.particle_number],
            h_field: vec![Vec3::default(); self.particle_number],
            pos_vel: vec![[Vec3::default(); 3]; self.particle_number],
            dir_vel: vec![[Vec3::default(); 2]; self.particle_number],

            param: self,
            particle_vol,
            rve_side_len,
            radius_eq,
            t_drag,
            r_drag,
            e_sus_x,
            e_sus_z,
            mag_dipole,

            log_dir,
        };
        this.update_el_dipoles();
        this
    }

    pub fn to_json(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn Error>> {
        let file = File::create(path)?;
        serde_json::ser::to_writer_pretty(file, &self)?;
        Ok(())
    }
}
