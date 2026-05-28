use rand::prelude::*;
use std::f32::consts::PI;
use std::path::Path;
use std::{error::Error, fs::File};

use super::{
    Simulation, Vec3,
    math::{Bounded, UnitVec},
};

const MEGA: f32 = 1e6;
const KILO: f32 = 1e3;
const MICRO: f32 = 1e-6;

pub trait Invalid {
    const INVALID: Self;
}
impl Invalid for f32 {
    const INVALID: Self = Self::NAN;
}
impl Invalid for f64 {
    const INVALID: Self = Self::NAN;
}
impl Invalid for Vec3 {
    const INVALID: Self = Vec3 {
        x: f32::NAN,
        y: f32::NAN,
        z: f32::NAN,
        _pad: 0.0,
    };
}

#[derive(Clone)]
pub enum ValueOrFn<T: Invalid> {
    Value(T),
    Fn(fn(f32) -> T),
}
impl<T: Invalid + serde::Serialize> serde::Serialize for ValueOrFn<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            // Serialize the actual value
            ValueOrFn::Value(v) => v.serialize(serializer),
            // Serialize the "Invalid" sentinel value (e.g., NaN)
            ValueOrFn::Fn(_) => T::INVALID.serialize(serializer),
        }
    }
}
impl<T: Invalid> From<T> for ValueOrFn<T> {
    fn from(value: T) -> Self {
        Self::Value(value)
    }
}

impl<T: Invalid> From<fn(f32) -> T> for ValueOrFn<T> {
    fn from(value: fn(f32) -> T) -> Self {
        Self::Fn(value)
    }
}

impl<T: Copy + Invalid> ValueOrFn<T> {
    pub fn get(&self, time: f32) -> T {
        match self {
            ValueOrFn::Value(v) => *v,
            ValueOrFn::Fn(f) => f(time),
        }
    }
}

#[derive(Clone, serde::Serialize)]
pub struct SimulationBuilder {
    pub fill_fraction: f32,
    pub particle_number: usize,
    pub small_saxis: f32,
    pub big_saxis: f32,

    pub mag_moment_density: f32,

    pub viscosity: ValueOrFn<f32>,

    pub epsilon_mat: f32,
    pub epsilon_part: f32,

    pub h_field_dir: ValueOrFn<Vec3>,
    pub h_field_norm: ValueOrFn<f32>,
    pub e_field_dir: ValueOrFn<Vec3>,
    pub e_field_norm: ValueOrFn<f32>,

    pub repulsion_factor: f32,
    pub velocity_factor: f32,

    pub duration: f32,

    pub log_frames: u32,
    pub seed: Option<u64>,
    pub name: String,
}

impl Default for SimulationBuilder {
    fn default() -> Self {
        use ValueOrFn::Value;
        let big_saxis: f32 = 2.5 * MICRO;
        let mag_moment_density: f32 = 380.0 * KILO;

        Self {
            fill_fraction: 0.01,
            particle_number: 500,
            small_saxis: big_saxis / 3.5,
            big_saxis,
            mag_moment_density: 380.0 * KILO,
            viscosity: Value(3.5),
            epsilon_mat: 2.0,
            epsilon_part: 10.0,
            h_field_dir: Value(Vec3::new(0.0, 0.0, 1.0)),
            h_field_norm: Value(5.0 * mag_moment_density),
            e_field_norm: Value(100.0 * MEGA),
            e_field_dir: Value(Vec3::new(0.0, 1.0, 0.0)),
            repulsion_factor: 40.0,
            duration: 3.0,
            velocity_factor: 0.5,
            log_frames: 50,
            seed: None,
            name: String::new(),
        }
    }
}

#[derive(Clone, serde::Serialize)]
pub struct SimulationParameters {
    pub fill_fraction: f32,
    pub particle_number: usize,
    pub small_saxis: f32,
    pub big_saxis: f32,

    pub mag_moment_density: f32,

    viscosity: ValueOrFn<f32>,

    pub epsilon_mat: f32,
    pub epsilon_part: f32,

    h_field_dir: ValueOrFn<Vec3>,
    h_field_norm: ValueOrFn<f32>,
    e_field_dir: ValueOrFn<Vec3>,
    e_field_norm: ValueOrFn<f32>,

    pub repulsion_factor: f32,
    pub velocity_factor: f32,

    pub duration: f32,

    pub log_frames: u32,
    pub seed: u64,
    pub name: String,

    pub particle_vol: f32,
    pub rve_side_len: f32,
    pub radius_eq: f32,
    pub e_sus_x: f32,
    pub e_sus_z: f32,
    pub mag_dipole: f32,
}

impl SimulationParameters {
    pub fn t_drag(&self, time: f32) -> f32 {
        6.0 * PI * self.viscosity.get(time) * self.radius_eq
    }
    pub fn r_drag(&self, time: f32) -> f32 {
        8.0 * PI * self.viscosity.get(time) * self.big_saxis.powi(2) * self.small_saxis
    }
    pub fn ext_e_field(&self, time: f32) -> Vec3 {
        self.e_field_norm.get(time) * self.e_field_dir.get(time)
    }
    pub fn ext_h_field(&self, time: f32) -> Vec3 {
        self.h_field_norm.get(time) * self.h_field_dir.get(time)
    }

    pub fn to_json(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn Error>> {
        let file = File::create(path)?;
        serde_json::ser::to_writer_pretty(file, &self)?;
        Ok(())
    }
}

impl Into<SimulationParameters> for SimulationBuilder {
    fn into(self) -> SimulationParameters {
        let particle_vol = 4.0 / 3.0 * PI * self.big_saxis.powi(2) * self.small_saxis;
        let rve_side_len =
            (particle_vol * self.particle_number as f32 / self.fill_fraction).powf(1.0 / 3.0);
        let radius_eq = (self.big_saxis.powi(2) * self.small_saxis).powf(1.0 / 3.0);
        let shape_factor = (self.big_saxis.powi(2) * self.small_saxis)
            / (2.0 * (self.big_saxis.powi(2) - self.small_saxis.powi(2)))
            * (PI / 2.0 / (self.big_saxis.powi(2) - self.small_saxis.powi(2)).sqrt()
                - self.small_saxis / self.big_saxis.powi(2));
        let e_sus_x = (self.epsilon_part - self.epsilon_mat)
            / (1.0 + (self.epsilon_part - self.epsilon_mat) / self.epsilon_mat * shape_factor);
        let e_sus_z = (self.epsilon_part - self.epsilon_mat)
            / (1.0
                + (self.epsilon_part - self.epsilon_mat) / self.epsilon_mat
                    * (1.0 - 2.0 * shape_factor));
        let mag_dipole = particle_vol * self.mag_moment_density;

        let seed = self.seed.unwrap_or_else(rand::random::<u64>);

        SimulationParameters {
            fill_fraction: self.fill_fraction,
            particle_number: self.particle_number,
            small_saxis: self.small_saxis,
            big_saxis: self.big_saxis,

            mag_moment_density: self.mag_moment_density,

            viscosity: self.viscosity,

            epsilon_mat: self.epsilon_mat,
            epsilon_part: self.epsilon_part,

            h_field_dir: self.h_field_dir,
            h_field_norm: self.h_field_norm,
            e_field_dir: self.e_field_dir,
            e_field_norm: self.e_field_norm,

            repulsion_factor: self.repulsion_factor,
            velocity_factor: self.velocity_factor,

            duration: self.duration,

            log_frames: self.log_frames,
            seed,
            name: self.name,
            particle_vol,
            rve_side_len,
            radius_eq,
            e_sus_x,
            e_sus_z,
            mag_dipole,
        }
    }
}

impl SimulationBuilder {
    pub fn build(self) -> Simulation {
        let params: SimulationParameters = self.into();
        let mut rng = rand::rngs::StdRng::seed_from_u64(params.seed);

        let mut positions: Vec<Vec3> = Vec::with_capacity(params.particle_number);
        let min_dist = 2.0 * params.radius_eq * 1.1; // 10% clearance
        let mut attempts = 0;
        while positions.len() < params.particle_number {
            let candidate: Vec3 = rng.sample(Bounded(params.rve_side_len));
            let overlap = positions.iter().any(|p| {
                let r = (candidate - *p) % params.rve_side_len;
                r.norm() < min_dist
            });
            if !overlap {
                positions.push(candidate);
            }
            attempts += 1;
            if attempts > params.particle_number * 10_000 {
                eprintln!(
                    "Warning: could not place all particles without overlap after {attempts} attempts"
                );
                break;
            }
        }

        let directions: Vec<Vec3> = (&mut rng)
            .sample_iter(UnitVec)
            .take(params.particle_number)
            .collect();
        let metal = crate::gpu::MetalState::new(&params, &positions, &directions);

        let this = Simulation { params, metal };
        this.update_el_dipoles();
        this
    }
}
