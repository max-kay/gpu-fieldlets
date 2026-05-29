use rand::prelude::*;
use std::f32::consts::PI;
use std::path::Path;
use std::{error::Error, fs::File};

use crate::gpu::{FrameSpec, GPUParams, Stage};

use super::{
    Simulation, Vec3,
    math::{Bounded, UnitVec},
};

const MEGA: f32 = 1e6;
const KILO: f32 = 1e3;
const MICRO: f32 = 1e-6;

const EPSILON_0: f32 = 8.8541878188e-12;
const MU_0: f32 = 1.25663706127e-6;

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

    pub seed: Option<u64>,

    pub name: String,
    pub log_frames: u32,
    pub camera: CameraBuilder,
}

const DEFAULT_MAG_MOMENT_DENSITY: f32 = 380.0 * KILO;
impl Default for SimulationBuilder {
    fn default() -> Self {
        use ValueOrFn::Value;
        let big_saxis: f32 = 2.5 * MICRO;

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
            h_field_norm: Value(5.0 * DEFAULT_MAG_MOMENT_DENSITY),
            e_field_norm: Value(100.0 * MEGA),
            e_field_dir: Value(Vec3::new(0.0, 1.0, 0.0)),
            repulsion_factor: 40.0,
            duration: 3.0,
            velocity_factor: 0.3,
            log_frames: 50,
            seed: None,
            name: String::new(),
            camera: CameraBuilder::default(),
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

    pub seed: u64,

    pub particle_vol: f32,
    pub rve_side_len: f32,
    pub radius_eq: f32,
    pub e_sus_x: f32,
    pub e_sus_z: f32,
    pub mag_dipole: f32,

    pub name: String,
    pub log_frames: u32,
    pub camera: CameraBuilder,
}

impl SimulationParameters {
    pub fn t_drag(&self, time: f32) -> f32 {
        6.0 * PI * self.viscosity.get(time) * self.radius_eq
    }
    pub fn r_drag(&self, time: f32) -> f32 {
        8.0 * PI * self.viscosity.get(time) * self.radius_eq.powi(3)
    }
    pub fn ext_e_field(&self, time: f32) -> Vec3 {
        self.e_field_norm.get(time) * self.e_field_dir.get(time)
    }
    pub fn ext_h_field(&self, time: f32) -> Vec3 {
        self.h_field_norm.get(time) * self.h_field_dir.get(time)
    }

    pub fn gpu_params(&self, time: f32) -> GPUParams {
        GPUParams {
            ext_h_field: self.ext_h_field(time),
            ext_e_field: self.ext_e_field(time),
            particle_number: self.particle_number as u32,
            h_field_prefactor: self.mag_dipole / (4.0 * PI) * self.rve_side_len.powi(3),
            e_field_prefactor: 1.0 / (4.0 * PI * EPSILON_0 * self.epsilon_mat)
                * self.rve_side_len.powi(3),
            left_dipole_prefactor: self.particle_vol * EPSILON_0 * self.e_sus_x,
            right_dipole_prefactor: self.particle_vol * EPSILON_0 * (self.e_sus_z - self.e_sus_x),
            h_force_prefactor: 3.0 * MU_0 * self.mag_dipole.powi(2) / (4.0 * PI)
                * self.rve_side_len.powi(4),
            e_force_prefactor: 3.0 / (EPSILON_0 * self.epsilon_mat * 2.0 * PI)
                * self.rve_side_len.powi(4),
            r_force_prefactor: 3.0 * MU_0 * self.mag_dipole.powi(2)
                / (2.0 * PI * (2.0 * self.radius_eq).powi(4))
                * self.rve_side_len.powi(4),
            h_torque_prefactor: MU_0 * self.mag_dipole,
            e_torque_prefactor: self.particle_vol * EPSILON_0 * (self.e_sus_z - self.e_sus_x),
            rve_side_len: self.rve_side_len,
            repulsion_factor: self.repulsion_factor,
            radius_eq: self.radius_eq,
            t_drag: self.t_drag(time),
            r_drag: self.r_drag(time),
        }
    }

    pub fn to_json(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn Error>> {
        let file = File::create(path)?;
        serde_json::ser::to_writer_pretty(file, &self)?;
        Ok(())
    }

    pub fn frame_spec(&self, time: f32) -> FrameSpec {
        let root = self.camera.root.get(time);
        let dist = root.norm();
        let scale =
            f32::sqrt(3.0) * 1.2 / (self.camera.dims[0].min(self.camera.dims[1]) as f32) / dist;
        let dir = -root.normalised();
        let u_s2 = -dir.cross(Vec3::new(0.0, 0.0, 1.0)).normalised();
        let u_s1 = dir.cross(u_s2).normalised();
        FrameSpec {
            dims: self.camera.dims,
            particle_number: self.particle_number as u32,
            oversamples: self.camera.oversamples,
            cam_root: root,
            cam_s1: u_s1 * scale,
            cam_s2: u_s2 * scale,
            cam_dir: dir,
            ell_axes: Vec3::new(
                self.big_saxis / self.rve_side_len,
                self.big_saxis / self.rve_side_len,
                self.small_saxis / self.rve_side_len,
            ),
            ell_color: self.camera.particle_color,
            light_dir: self.camera.light_dir,
            bg_color: self.camera.background,
            ambient_light: self.camera.ambient_light,
        }
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
            camera: self.camera,
        }
    }
}

impl SimulationBuilder {
    pub fn build(self) -> Simulation {
        let params: SimulationParameters = self.into();
        let mut rng = rand::rngs::StdRng::seed_from_u64(params.seed);

        let mut positions: Vec<Vec3> = Vec::with_capacity(params.particle_number);
        let clearance = 1.1;
        let min_dist = 2.0 * params.radius_eq * clearance / params.rve_side_len;
        let mut attempts = 0;
        while positions.len() < params.particle_number {
            let candidate: Vec3 = rng.sample(Bounded(1.0));
            let overlap = positions.iter().any(|p| {
                let r = (candidate - *p) % 1.0;
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

        let gpu_params = params.gpu_params(0.0);
        metal.run_stage(Stage::ElDipoles, &gpu_params);
        Simulation { params, metal }
    }
}

#[derive(Clone, serde::Serialize)]
pub struct CameraBuilder {
    pub dims: [u32; 2],
    pub root: ValueOrFn<Vec3>,
    pub oversamples: u32,
    pub background: Vec3,
    pub particle_color: Vec3,
    pub light_dir: Vec3,
    pub ambient_light: f32,
}

impl Default for CameraBuilder {
    fn default() -> Self {
        use ValueOrFn::Value;
        Self {
            dims: [1000, 800],
            root: Value(3.0 * Vec3::new(1.0, 0.0, 0.0)),
            oversamples: 3,
            background: Vec3::new(1.0, 1.0, 1.0),
            particle_color: Vec3::new(0.3, 0.12, 0.8),
            light_dir: Vec3::new(-0.8, 0.4, 2.5).normalised(),
            ambient_light: 0.8,
        }
    }
}
