use rand::prelude::*;
use std::f64::consts::PI;
use std::path::Path;
use std::{error::Error, fs::File};

use crate::gpu::{FrameSpec, GpuParams, Stage};

use super::{
    Simulation, Vec3,
    math::{Bounded, UnitVec},
};

const MEGA: f64 = 1e6;
const KILO: f64 = 1e3;
const MICRO: f64 = 1e-6;

const EPSILON_0: f64 = 8.8541878188e-12;
const MU_0: f64 = 1.25663706127e-6;

const H_REF: f64 = 1.9 * MEGA;
const E_REF: f64 = 100.0 * MEGA;

pub trait Invalid {
    const INVALID: Self;
}
impl Invalid for f64 {
    const INVALID: Self = Self::NAN;
}

impl Invalid for f32 {
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
    Fn(fn(f64) -> T),
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

impl<T: Invalid> From<fn(f64) -> T> for ValueOrFn<T> {
    fn from(value: fn(f64) -> T) -> Self {
        Self::Fn(value)
    }
}

impl<T: Copy + Invalid> ValueOrFn<T> {
    pub fn get(&self, time: f64) -> T {
        match self {
            ValueOrFn::Value(v) => *v,
            ValueOrFn::Fn(f) => f(time),
        }
    }
}

#[derive(Clone, serde::Serialize)]
pub struct SimulationBuilder {
    pub fill_fraction: f64,
    pub particle_number: usize,
    pub small_saxis: f64,
    pub big_saxis: f64,

    pub mag_moment_density: f64,

    pub viscosity: f64,

    pub epsilon_mat: f64,
    pub epsilon_part: f64,

    pub h_field_dir: ValueOrFn<Vec3>,
    pub h_field_norm: ValueOrFn<f64>,
    pub e_field_dir: ValueOrFn<Vec3>,
    pub e_field_norm: ValueOrFn<f64>,

    pub repulsion_factor: f64,
    pub velocity_factor: f64,

    pub duration: f64,

    pub seed: Option<u64>,

    pub name: String,
    pub log_frames: u32,
    pub camera: CameraBuilder,
}

const DEFAULT_MAG_MOMENT_DENSITY: f64 = 380.0 * KILO;

impl Default for SimulationBuilder {
    fn default() -> Self {
        use ValueOrFn::Value;
        let big_saxis: f64 = 2.5 * MICRO;

        Self {
            fill_fraction: 0.01,
            particle_number: 500,
            small_saxis: big_saxis / 3.5,
            big_saxis,
            mag_moment_density: 380.0 * KILO,
            viscosity: 3.5,
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
    pub fill_fraction: f64,
    pub particle_number: usize,
    pub small_saxis: f64,
    pub big_saxis: f64,

    pub mag_moment_density: f64,
    pub char_time_scale: f64,

    pub t_drag: f64,
    pub r_drag: f64,

    pub epsilon_mat: f64,
    pub epsilon_part: f64,

    h_field_dir: ValueOrFn<Vec3>,
    h_field_norm: ValueOrFn<f64>,
    e_field_dir: ValueOrFn<Vec3>,
    e_field_norm: ValueOrFn<f64>,

    pub repulsion_factor: f64,
    pub velocity_factor: f64,

    pub duration: f64,

    pub seed: u64,

    pub particle_vol: f64,
    pub rve_side_len: f64,
    pub radius_eq: f64,
    pub e_sus_x: f64,
    pub e_sus_z: f64,
    pub mag_dipole: f64,

    pub name: String,
    pub log_frames: u32,
    pub camera: CameraBuilder,
}

impl SimulationParameters {
    pub fn ext_h_field(&self, time: f64) -> Vec3 {
        (self.h_field_norm.get(time) / H_REF) as f32 * self.h_field_dir.get(time)
    }
    pub fn ext_e_field(&self, time: f64) -> Vec3 {
        (self.e_field_norm.get(time) / E_REF) as f32 * self.e_field_dir.get(time)
    }

    pub fn gpu_params(&self, time: f64) -> GpuParams {
        let factor_1 =
            3.0 * MU_0 * self.mag_dipole.powi(2) / (4.0 * PI * self.rve_side_len.powi(4));
        let factor_2 = self.char_time_scale / (self.t_drag * self.rve_side_len);
        let h_force_prefactor = (factor_1 * factor_2) as f32;

        let factor_1 = 3.0 / (EPSILON_0 * self.epsilon_mat * 2.0 * PI);
        let factor_2 = self.char_time_scale / (self.t_drag * self.rve_side_len);
        let factor_3 = (E_REF * self.particle_vol * EPSILON_0 * self.e_sus_x).powi(2)
            / self.rve_side_len.powi(4);
        let e_force_prefactor = (factor_1 * factor_2 * factor_3) as f32;

        let factor_1 =
            3.0 * MU_0 * self.mag_dipole.powi(2) / (2.0 * PI * (2.0 * self.radius_eq).powi(4));
        let factor_2 = self.char_time_scale / (self.t_drag * self.rve_side_len);
        let r_force_prefactor = (factor_1 * factor_2) as f32;
        GpuParams {
            ext_h_field: self.ext_h_field(time),
            ext_e_field: self.ext_e_field(time),

            particle_number: self.particle_number as u32,
            h_field_prefactor: (1.0 / H_REF * self.mag_dipole
                / (4.0 * PI * self.rve_side_len.powi(3))) as f32,
            e_field_prefactor: (self.particle_vol * self.e_sus_x
                / (4.0 * PI * self.epsilon_mat * self.rve_side_len.powi(3)))
                as f32,
            right_dipole_prefactor: ((self.e_sus_z - self.e_sus_x) / self.e_sus_x) as f32,

            e_force_prefactor,
            r_force_prefactor,
            h_torque_prefactor: (self.char_time_scale / self.r_drag
                * MU_0
                * self.mag_dipole
                * H_REF) as f32,
            e_torque_prefactor: (self.char_time_scale / self.r_drag
                * self.particle_vol
                * EPSILON_0
                * (self.e_sus_z - self.e_sus_x)
                * E_REF.powi(2)) as f32,

            rve_side_len: self.rve_side_len as f32,
            repulsion_factor: self.repulsion_factor as f32,
            radius_eq: self.radius_eq as f32,
            h_force_prefactor,
        }
    }

    pub fn to_json(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn Error>> {
        let file = File::create(path)?;
        serde_json::ser::to_writer_pretty(file, &self)?;
        Ok(())
    }

    pub fn frame_spec(&self, time: f64) -> FrameSpec {
        let root = self.camera.root.get(time);
        let dist = root.norm();
        let aspect = self.camera.dims[0] as f32 / self.camera.dims[1] as f32;

        let dir = -root.normalised();
        let u_s1 = dir.cross(Vec3::new(0.0, 0.0, 1.0)).normalised();
        let u_s2 = u_s1.cross(dir).normalised();

        let sphere_radius = 0.5 * f32::sqrt(3.0);
        let padding = 1.8;

        let half_fov_y = (sphere_radius * padding / dist).asin();

        let half_height = half_fov_y.tan();
        let half_width = half_height * aspect;

        let center_to_corner2 = padding.powi(2) / 4.0 * (1.0 + aspect.powi(2));
        let discriminant =
            (8.0 * center_to_corner2 + 2.0).powi(2) - 4.0 * (4.0 * center_to_corner2 - 1.0).powi(2);

        let extra_padding = 1.1;
        let sub_window_ratio = (8.0 * center_to_corner2 + 2.0 - f32::sqrt(discriminant))
            / (8.0 * center_to_corner2 - 2.0)
            / extra_padding;
        let sub_screen_width = (self.camera.dims[0] as f32 * sub_window_ratio) as u32;
        let sub_screen_height = (self.camera.dims[1] as f32 * sub_window_ratio) as u32;

        FrameSpec {
            dims: self.camera.dims,
            sub_img_dims: [sub_screen_width, sub_screen_height],

            particle_number: self.particle_number as u32,
            oversamples: self.camera.oversamples,
            ambient_light: (self.camera.ambient_light) as f32,
            culling_radius: 1.0 / padding,

            cam_root: root,
            cam_s1: u_s1 * half_width,
            cam_s2: u_s2 * half_height,
            cam_dir: dir,
            ell_axes: Vec3::new(
                (self.big_saxis / self.rve_side_len) as f32,
                (self.big_saxis / self.rve_side_len) as f32,
                (self.small_saxis / self.rve_side_len) as f32,
            ),
            light_dir: self.camera.light_dir,
            h_field: self.h_field_dir.get(time),
            e_field: self.ext_e_field(time),
        }
    }
}

impl Into<SimulationParameters> for SimulationBuilder {
    fn into(self) -> SimulationParameters {
        let particle_vol = 4.0 / 3.0 * PI * self.big_saxis.powi(2) * self.small_saxis;
        let rve_side_len =
            (particle_vol * self.particle_number as f64 / self.fill_fraction).powf(1.0 / 3.0);
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
        let t_drag = 6.0 * PI * self.viscosity * radius_eq;

        SimulationParameters {
            fill_fraction: self.fill_fraction,
            particle_number: self.particle_number,
            small_saxis: self.small_saxis,
            big_saxis: self.big_saxis,

            char_time_scale: 4.0 * PI * t_drag * rve_side_len * (2.0 * radius_eq).powi(4)
                / (3.0 * MU_0 * mag_dipole.powi(2)),

            mag_moment_density: self.mag_moment_density,

            t_drag,
            r_drag: 8.0 * PI * self.viscosity * radius_eq.powi(3),

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
                r.norm() < min_dist as f32
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
        let gpu_state = crate::gpu::GpuState::new(&params, &positions, &directions);

        let gpu_params = params.gpu_params(0.0);
        let pass = gpu_state.begin_pass(&gpu_params, None);
        pass.dispatch(Stage::EDipole);
        pass.commit_and_wait();
        Simulation { params, gpu_state }
    }
}

#[derive(Clone, serde::Serialize)]
pub struct CameraBuilder {
    pub dims: [u32; 2],
    pub root: ValueOrFn<Vec3>,
    pub oversamples: u32,
    pub light_dir: Vec3,
    pub ambient_light: f64,
}

impl Default for CameraBuilder {
    fn default() -> Self {
        use ValueOrFn::Value;
        Self {
            dims: [1000, 800],
            root: Value(3.0 * Vec3::new(1.0, 1.0, 1.0)),
            oversamples: 3,
            light_dir: Vec3::new(-0.8, 0.4, 2.5).normalised(),
            ambient_light: 0.8,
        }
    }
}
