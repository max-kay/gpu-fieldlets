use std::path::Path;
use std::{error::Error, fs::File};

use chrono::{self, Local};
use rand::prelude::*;

use super::{
    Float, PI, Simulation, Vec3,
    math::{Bounded, UnitVec},
};

const MEGA: Float = 1e6;
const KILO: Float = 1e3;
const NANO: Float = 1e-9;

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
        x: Float::NAN,
        y: Float::NAN,
        z: Float::NAN,
    };
}

#[derive(Clone)]
pub enum ValueOrFn<T: Invalid> {
    Value(T),
    Fn(fn(Float) -> T),
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

impl<T: Invalid> From<fn(Float) -> T> for ValueOrFn<T> {
    fn from(value: fn(Float) -> T) -> Self {
        Self::Fn(value)
    }
}

impl<T: Copy + Invalid> ValueOrFn<T> {
    pub fn get(&self, time: Float) -> T {
        match self {
            ValueOrFn::Value(v) => *v,
            ValueOrFn::Fn(f) => f(time),
        }
    }
}

#[derive(Clone, serde::Serialize)]
pub struct SimulationBuilder {
    pub fill_fraction: Float,
    pub particle_number: usize,
    pub small_saxis: Float,
    pub big_saxis: Float,

    pub mag_moment_density: Float,

    pub viscosity: ValueOrFn<Float>,

    pub epsilon_mat: Float,
    pub epsilon_part: Float,

    pub h_field_dir: ValueOrFn<Vec3>,
    pub h_field_norm: ValueOrFn<Float>,
    pub e_field_dir: ValueOrFn<Vec3>,
    pub e_field_norm: ValueOrFn<Float>,

    pub repulsion_factor: Float,

    pub delta_time: Float,
    pub duration: Float,

    pub log_step: usize,
    pub seed: Option<u64>,
    pub name: String,
}

impl Default for SimulationBuilder {
    fn default() -> Self {
        use ValueOrFn::Value;
        let small_saxis: Float = 75.0 * NANO;
        let mag_moment_density: Float = 380.0 * KILO;

        Self {
            fill_fraction: 0.01,
            particle_number: 1000,
            small_saxis: 75.0 * NANO,
            big_saxis: small_saxis * 3.5,
            mag_moment_density: 380.0 * KILO,
            viscosity: Value(3.5),
            epsilon_mat: 2.0,
            epsilon_part: 10.0,
            h_field_dir: Value(Vec3::new(0.0, 0.0, 1.0)),
            h_field_norm: Value(5.0 * mag_moment_density),
            e_field_norm: Value(100.0 * MEGA),
            e_field_dir: Value(Vec3::new(0.0, 1.0, 0.0)),
            repulsion_factor: 40.0,
            delta_time: 0.00001,
            duration: 1.0,
            log_step: 1000,
            seed: None,
            name: String::new(),
        }
    }
}

#[derive(Clone, serde::Serialize)]
pub struct SimulationParameters {
    pub fill_fraction: Float,
    pub particle_number: usize,
    pub small_saxis: Float,
    pub big_saxis: Float,

    pub mag_moment_density: Float,

    viscosity: ValueOrFn<Float>,

    pub epsilon_mat: Float,
    pub epsilon_part: Float,

    h_field_dir: ValueOrFn<Vec3>,
    h_field_norm: ValueOrFn<Float>,
    e_field_dir: ValueOrFn<Vec3>,
    e_field_norm: ValueOrFn<Float>,

    pub repulsion_factor: Float,

    pub delta_time: Float,
    pub duration: Float,

    pub log_step: usize,
    pub seed: u64,
    pub name: String,

    pub particle_vol: Float,
    pub rve_side_len: Float,
    pub radius_eq: Float,
    pub e_sus_x: Float,
    pub e_sus_z: Float,
    pub mag_dipole: Float,
}

impl SimulationParameters {
    pub fn t_drag(&self, time: Float) -> Float {
        6.0 * PI * self.viscosity.get(time) * self.radius_eq
    }
    pub fn r_drag(&self, time: Float) -> Float {
        8.0 * PI * self.viscosity.get(time) * self.big_saxis.powi(2) * self.small_saxis
    }
    pub fn ext_e_field(&self, time: Float) -> Vec3 {
        self.e_field_norm.get(time) * self.e_field_dir.get(time)
    }
    pub fn ext_h_field(&self, time: Float) -> Vec3 {
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
            (particle_vol * self.particle_number as Float / self.fill_fraction).powf(1.0 / 3.0);
        let radius_eq = (self.big_saxis.powi(2) * self.small_saxis).powf(1.0 / 3.0);
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

            delta_time: self.delta_time,
            duration: self.duration,

            log_step: self.log_step,
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

        let log_dir = format!("out/{}", Local::now().format("%Y-%m-%d_%H-%M-%S"));
        if let Err(err) = std::fs::create_dir_all(&log_dir) {
            eprintln!("could not make log dir: {err}")
        }

        if let Err(err) = params.to_json(format!("{}/config.json", log_dir)) {
            eprintln!("could not log configuration: {err}")
        }

        let mut this = Simulation {
            positions,
            directions: (&mut rng)
                .sample_iter(UnitVec)
                .take(params.particle_number)
                .collect(),
            el_dipole_moments: vec![Vec3::default(); params.particle_number],
            e_field: vec![params.ext_e_field(0.0); params.particle_number],
            h_field: vec![Vec3::default(); params.particle_number],
            pos_vel: vec![Vec3::default(); params.particle_number],
            dir_vel: vec![Vec3::default(); params.particle_number],

            params,
            log_dir,
        };
        this.update_el_dipoles();
        this
    }
}
