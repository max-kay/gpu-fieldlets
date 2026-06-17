use std::mem;
use std::ptr::NonNull;

use dispatch2::DispatchData;
use image::{ImageBuffer, ImageResult, Rgba};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSString, NSUInteger};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};

use crate::math::Vec3;
use crate::numpy::Numpy;
use crate::params::SimulationParameters;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct GPUParams {
    pub ext_h_field: Vec3,
    pub ext_e_field: Vec3,

    pub particle_number: u32,
    pub h_field_prefactor: f32,
    pub e_field_prefactor: f32,
    pub left_dipole_prefactor: f32,

    pub right_dipole_prefactor: f32,
    pub h_force_prefactor: f32,
    pub e_force_prefactor: f32,
    pub r_force_prefactor: f32,

    pub h_torque_prefactor: f32,
    pub e_torque_prefactor: f32,
    pub rve_side_len: f32,
    pub repulsion_factor: f32,

    pub radius_eq: f32,
    pub t_drag: f32,
    pub r_drag: f32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct FrameSpec {
    pub dims: [u32; 2],
    pub particle_number: u32,
    pub oversamples: u32,
    pub cam_root: Vec3,
    pub cam_s1: Vec3,
    pub cam_s2: Vec3,
    pub cam_dir: Vec3,
    pub ell_axes: Vec3,
    pub ell_color: Vec3,
    pub light_dir: Vec3,
    pub bg_color: Vec3,
    pub ambient_light: f32,
}

pub enum Stage {
    EField,
    HField,
    EDipole,
    Velocity,
    Position(f32),
    Direction(f32),
}

pub struct MetalState {
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline_e_field: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pipeline_h_field: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pipeline_e_dipole: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pipeline_velocity: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pipeline_position: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pipeline_direction: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pipeline_check: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pipeline_render: Retained<ProtocolObject<dyn MTLComputePipelineState>>,

    buf_position: Retained<ProtocolObject<dyn MTLBuffer>>,
    buf_direction: Retained<ProtocolObject<dyn MTLBuffer>>,
    buf_e_dipole: Retained<ProtocolObject<dyn MTLBuffer>>,
    buf_e_field: Retained<ProtocolObject<dyn MTLBuffer>>,
    buf_h_field: Retained<ProtocolObject<dyn MTLBuffer>>,
    buf_velocity: Retained<ProtocolObject<dyn MTLBuffer>>,
    buf_check_output: Retained<ProtocolObject<dyn MTLBuffer>>,

    buf_img: Retained<ProtocolObject<dyn MTLBuffer>>,
}

const SHADER_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders.metallib"));

impl MetalState {
    pub fn new(params: &SimulationParameters, positions: &[Vec3], directions: &[Vec3]) -> Self {
        let device = MTLCreateSystemDefaultDevice().expect("no metal device");
        let queue = device.newCommandQueue().unwrap();

        let library = device
            .newLibraryWithData_error(DispatchData::from_static_bytes(SHADER_BYTES).as_ref())
            .expect("failed to load embedded metal library");

        let get_pipeline = |name: &str| {
            let name_ns = NSString::from_str(name);
            let function = library
                .newFunctionWithName(&name_ns)
                .expect(&format!("function `{}` not found", name));
            device
                .newComputePipelineStateWithFunction_error(&function)
                .unwrap()
        };

        let ext_e = vec![params.ext_e_field(0.0); params.particle_number];

        unsafe {
            Self {
                queue,
                pipeline_e_field: get_pipeline("update_e_field"),
                pipeline_h_field: get_pipeline("update_h_field"),
                pipeline_e_dipole: get_pipeline("update_e_dipole"),
                pipeline_velocity: get_pipeline("update_velocity"),
                pipeline_position: get_pipeline("update_position"),
                pipeline_direction: get_pipeline("update_direction"),
                pipeline_check: get_pipeline("check_finite_and_max_vel"),
                pipeline_render: get_pipeline("render_kernel"),

                buf_position: device
                    .newBufferWithBytes_length_options(
                        NonNull::new(positions.as_ptr() as *mut _).unwrap(),
                        mem::size_of_val(positions),
                        MTLResourceOptions::StorageModeShared,
                    )
                    .unwrap(),
                buf_direction: device
                    .newBufferWithBytes_length_options(
                        NonNull::new(directions.as_ptr() as *mut _).unwrap(),
                        mem::size_of_val(directions),
                        MTLResourceOptions::StorageModeShared,
                    )
                    .unwrap(),
                buf_e_dipole: device
                    .newBufferWithLength_options(
                        mem::size_of::<Vec3>() * params.particle_number,
                        MTLResourceOptions::StorageModeShared,
                    )
                    .unwrap(),
                buf_e_field: device
                    .newBufferWithBytes_length_options(
                        NonNull::new(ext_e.as_ptr() as *mut _).unwrap(),
                        mem::size_of_val(&*ext_e),
                        MTLResourceOptions::StorageModeShared,
                    )
                    .unwrap(),
                buf_h_field: device
                    .newBufferWithLength_options(
                        mem::size_of::<Vec3>() * params.particle_number,
                        MTLResourceOptions::StorageModeShared,
                    )
                    .unwrap(),
                buf_velocity: device
                    .newBufferWithLength_options(
                        mem::size_of::<Vec3>() * params.particle_number,
                        MTLResourceOptions::StorageModeShared,
                    )
                    .unwrap(),
                buf_check_output: device
                    .newBufferWithLength_options(
                        2 * mem::size_of::<f32>(),
                        MTLResourceOptions::StorageModeShared,
                    )
                    .unwrap(),

                buf_img: device
                    .newBufferWithLength_options(
                        mem::size_of::<u32>()
                            * (params.camera.dims[0] * params.camera.dims[1]) as NSUInteger,
                        MTLResourceOptions::StorageModeShared,
                    )
                    .unwrap(),
            }
        }
    }
}

fn force_bracket_term(rji: Vec3, di: Vec3, dj: Vec3) -> Vec3 {
    let f1_dot = di.dot(dj) - 5.0 * rji.dot(dj) * rji.dot(di);
    let f1 = f1_dot * rji;
    let f2 = rji.dot(di) * dj + rji.dot(dj) * di;
    return f1 + f2;
}

fn field_bracket_term(rji: Vec3, dj: Vec3) -> Vec3 {
    3.0 * dj.dot(rji) * rji - dj
}

#[allow(dead_code)]
impl MetalState {
    pub fn update_e_dipole(&self, params: &GPUParams) {
        unsafe {
            let e_field = std::slice::from_raw_parts(
                self.buf_e_field.contents().as_ptr() as *const Vec3,
                params.particle_number as usize,
            );
            let directions = std::slice::from_raw_parts(
                self.buf_direction.contents().as_ptr() as *const Vec3,
                params.particle_number as usize,
            );
            let e_dipole = std::slice::from_raw_parts_mut(
                self.buf_e_dipole.contents().as_mut() as *mut _ as *mut Vec3,
                params.particle_number as usize,
            );
            for ((e_i, d_i), p_i) in e_field
                .iter()
                .zip(directions.iter())
                .zip(e_dipole.iter_mut())
            {
                *p_i = params.left_dipole_prefactor * *e_i
                    + params.right_dipole_prefactor * d_i.dot(*e_i) * *d_i
            }
        }
    }

    pub fn update_e_field(&self, params: &GPUParams) {
        unsafe {
            let pos = std::slice::from_raw_parts(
                self.buf_position.contents().as_ptr() as *const Vec3,
                params.particle_number as usize,
            );
            let e_dipole = std::slice::from_raw_parts(
                self.buf_e_dipole.contents().as_ptr() as *const Vec3,
                params.particle_number as usize,
            );
            let e_field = std::slice::from_raw_parts_mut(
                self.buf_e_field.contents().as_mut() as *mut _ as *mut Vec3,
                params.particle_number as usize,
            );
            for i in 0..(params.particle_number as usize) {
                let mut e_field_i = params.ext_e_field;
                for j in 0..(params.particle_number as usize) {
                    if i == j {
                        continue;
                    }
                    let r_ji = (*pos.get_unchecked(i) - *pos.get_unchecked(j)) % 1.0;
                    let dist = r_ji.norm();
                    let r_ji_hat = r_ji / dist;
                    e_field_i += params.e_field_prefactor / dist.powi(3)
                        * field_bracket_term(r_ji_hat, *e_dipole.get_unchecked(j));
                }
                *e_field.get_unchecked_mut(i) = e_field_i;
            }
        }
    }

    pub fn update_h_field(&self, params: &GPUParams) {
        unsafe {
            let pos = std::slice::from_raw_parts(
                self.buf_position.contents().as_ptr() as *const Vec3,
                params.particle_number as usize,
            );
            let dir = std::slice::from_raw_parts(
                self.buf_direction.contents().as_ptr() as *const Vec3,
                params.particle_number as usize,
            );
            let h_field = std::slice::from_raw_parts_mut(
                self.buf_h_field.contents().as_mut() as *mut _ as *mut Vec3,
                params.particle_number as usize,
            );
            for i in 0..(params.particle_number as usize) {
                let mut h_field_i = params.ext_e_field;
                for j in 0..(params.particle_number as usize) {
                    if i == j {
                        continue;
                    }
                    let r_ji = (*pos.get_unchecked(i) - *pos.get_unchecked(j)) % 1.0;
                    let dist = r_ji.norm();
                    let r_ji_hat = r_ji / dist;
                    h_field_i += params.h_field_prefactor / dist.powi(3)
                        * field_bracket_term(r_ji_hat, *dir.get_unchecked(j));
                }
                *h_field.get_unchecked_mut(i) = h_field_i;
            }
        }
    }

    pub fn update_velocity(&self, params: &GPUParams) {
        unsafe {
            let poss = std::slice::from_raw_parts(
                self.buf_position.contents().as_ptr() as *const Vec3,
                params.particle_number as usize,
            );
            let dirs = std::slice::from_raw_parts(
                self.buf_direction.contents().as_ptr() as *const Vec3,
                params.particle_number as usize,
            );
            let el_dipoles = std::slice::from_raw_parts(
                self.buf_e_dipole.contents().as_ptr() as *const Vec3,
                params.particle_number as usize,
            );
            let vels = std::slice::from_raw_parts_mut(
                self.buf_velocity.contents().as_mut() as *mut _ as *mut Vec3,
                params.particle_number as usize,
            );
            vels.iter_mut().for_each(|v| *v = Vec3::new(0.0, 0.0, 0.0));
            for i in 0..(params.particle_number as usize) {
                for j in (i + 1)..(params.particle_number as usize) {
                    let r_ji = (*poss.get_unchecked(i) - *poss.get_unchecked(j)) % 1.0;
                    let dist = r_ji.norm();
                    let r_ji_hat = r_ji / dist;

                    let f_h = params.h_force_prefactor / dist.powi(4)
                        * force_bracket_term(
                            r_ji_hat,
                            *dirs.get_unchecked(i),
                            *dirs.get_unchecked(j),
                        );

                    let f_e = params.e_force_prefactor / dist.powi(4)
                        * force_bracket_term(
                            r_ji_hat,
                            *el_dipoles.get_unchecked(i),
                            *el_dipoles.get_unchecked(j),
                        );

                    let exponent = -params.repulsion_factor
                        * (dist * params.rve_side_len / (2.0 * params.radius_eq) - 1.0);
                    let f_r = params.r_force_prefactor * (exponent.exp() * r_ji_hat);

                    let f_tot = f_h + f_e + f_r;
                    *vels.get_unchecked_mut(i) += f_tot / params.t_drag;
                    *vels.get_unchecked_mut(j) += -f_tot / params.t_drag;
                }
            }
        }
    }

    pub fn find_max_velocity(&self, params: &GPUParams) -> f32 {
        unsafe {
            let vels = std::slice::from_raw_parts(
                self.buf_velocity.contents().as_ptr() as *const Vec3,
                params.particle_number as usize,
            );
            let mut max_vel = 0.0;
            for v in vels {
                let v_norm = v.norm();
                if v_norm > max_vel {
                    max_vel = v_norm;
                }
            }
            max_vel
        }
    }

    pub fn update_position(&mut self, params: &GPUParams, delta_t: f32) {
        unsafe {
            let poss = std::slice::from_raw_parts_mut(
                self.buf_position.contents().as_mut() as *mut _ as *mut Vec3,
                params.particle_number as usize,
            );
            let vels = std::slice::from_raw_parts(
                self.buf_velocity.contents().as_ptr() as *const Vec3,
                params.particle_number as usize,
            );
            for (pos, vel) in poss.iter_mut().zip(vels.iter()) {
                *pos = (*pos + delta_t * *vel) % 1.0;
            }
        }
    }

    pub fn update_direction(&mut self, params: &GPUParams, delta_t: f32) {
        unsafe {
            let direction = std::slice::from_raw_parts_mut(
                self.buf_direction.contents().as_mut() as *mut _ as *mut Vec3,
                params.particle_number as usize,
            );
            let h_field = std::slice::from_raw_parts(
                self.buf_h_field.contents().as_ptr() as *const Vec3,
                params.particle_number as usize,
            );
            let e_field = std::slice::from_raw_parts(
                self.buf_e_field.contents().as_ptr() as *const Vec3,
                params.particle_number as usize,
            );

            for ((e_i, h_i), d_i) in e_field.iter().zip(h_field.iter()).zip(direction.iter_mut()) {
                let magnetic = params.h_torque_prefactor * (*h_i - *d_i * h_i.dot(*d_i));
                let electric =
                    params.e_torque_prefactor * e_i.dot(*d_i) * (*e_i - *d_i * e_i.dot(*d_i));

                let dir_vel = (magnetic + electric) / params.r_drag;

                *d_i = (*d_i + delta_t * dir_vel).normalised();
            }
        }
    }
}

impl MetalState {
    pub fn run_stage(&self, stage: Stage, params: &GPUParams) {
        let (pipeline, buffers, delta_t) = match stage {
            Stage::EField => (
                &*self.pipeline_e_field,
                vec![&*self.buf_position, &*self.buf_e_dipole, &*self.buf_e_field],
                None,
            ),
            Stage::HField => (
                &*self.pipeline_h_field,
                vec![
                    &*self.buf_position,
                    &*self.buf_direction,
                    &*self.buf_h_field,
                ],
                None,
            ),
            Stage::EDipole => (
                &*self.pipeline_e_dipole,
                vec![
                    &*self.buf_e_field,
                    &*self.buf_direction,
                    &*self.buf_e_dipole,
                ],
                None,
            ),
            Stage::Velocity => (
                &*self.pipeline_velocity,
                vec![
                    &*self.buf_position,
                    &*self.buf_direction,
                    &*self.buf_e_dipole,
                    &*self.buf_velocity,
                ],
                None,
            ),
            Stage::Position(dt) => (
                &*self.pipeline_position,
                vec![&*self.buf_position, &*self.buf_velocity],
                Some(dt),
            ),
            Stage::Direction(dt) => (
                &*self.pipeline_direction,
                vec![&*self.buf_direction, &*self.buf_h_field, &*self.buf_e_field],
                Some(dt),
            ),
        };
        let command_buffer = self.queue.commandBuffer().unwrap();
        let encoder = command_buffer.computeCommandEncoder().unwrap();

        encoder.setComputePipelineState(pipeline);

        unsafe {
            for (i, buf) in buffers.iter().enumerate() {
                encoder.setBuffer_offset_atIndex(Some(*buf), 0, i as _);
            }

            let mut next_index = buffers.len();
            if let Some(dt) = delta_t {
                encoder.setBytes_length_atIndex(
                    NonNull::new(&dt as *const f32 as *mut _).unwrap(),
                    mem::size_of::<f32>(),
                    next_index as _,
                );
                next_index += 1;
            }

            encoder.setBytes_length_atIndex(
                NonNull::new(params as *const _ as *mut _).unwrap(),
                mem::size_of::<GPUParams>(),
                next_index as _,
            );
        }

        let grid_size = MTLSize {
            width: params.particle_number as usize,
            height: 1,
            depth: 1,
        };
        let thread_group_size = MTLSize {
            width: pipeline
                .maxTotalThreadsPerThreadgroup()
                .min(params.particle_number as usize),
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreads_threadsPerThreadgroup(grid_size, thread_group_size);
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
    }

    pub fn run_max_and_check(&self, params: &GPUParams) -> (f32, bool) {
        let command_buffer = self.queue.commandBuffer().unwrap();
        let encoder = command_buffer.computeCommandEncoder().unwrap();

        encoder.setComputePipelineState(&self.pipeline_check);
        let buffers = [
            &*self.buf_e_dipole,
            &*self.buf_e_field,
            &*self.buf_h_field,
            &*self.buf_position,
            &*self.buf_direction,
            &*self.buf_velocity,
            &*self.buf_check_output,
        ];
        unsafe {
            for (i, buf) in buffers.iter().enumerate() {
                encoder.setBuffer_offset_atIndex(Some(*buf), 0, i as _);
            }
            encoder.setBytes_length_atIndex(
                NonNull::new(params as *const _ as *mut _).unwrap(),
                mem::size_of::<GPUParams>(),
                buffers.len() as _,
            );
        }

        let grid_size = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let thread_group_size = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreads_threadsPerThreadgroup(grid_size, thread_group_size);

        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        let output = unsafe {
            std::slice::from_raw_parts(self.buf_check_output.contents().as_ptr() as *const f32, 2)
        };
        (output[0], output[1] > 0.5)
    }

    pub fn run_plotting(&self, name: &str, dir: &str, spec: &FrameSpec) -> ImageResult<()> {
        let command_buffer = self.queue.commandBuffer().unwrap();
        let encoder = command_buffer.computeCommandEncoder().unwrap();
        encoder.setComputePipelineState(&self.pipeline_render);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&*self.buf_img), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&*self.buf_position), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&*self.buf_direction), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::new(spec as *const _ as *mut _).unwrap(),
                mem::size_of::<FrameSpec>(),
                3,
            );
        }

        let threads_per_grid = MTLSize {
            width: spec.dims[0] as _,
            height: spec.dims[1] as _,
            depth: 1,
        };
        let threads_per_threadgroup = MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        };
        encoder.dispatchThreads_threadsPerThreadgroup(threads_per_grid, threads_per_threadgroup);
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        let buffer = unsafe {
            std::slice::from_raw_parts(
                self.buf_img.contents().as_ptr() as *const u8,
                (spec.dims[0] * spec.dims[1]) as usize * mem::size_of::<u32>(),
            )
        };
        let img_buffer = ImageBuffer::<Rgba<u8>, _>::from_raw(spec.dims[0], spec.dims[1], buffer)
            .expect("Buffer size does not match the specified width and height");

        img_buffer.save(&format!("{}/frame_{}.png", dir, name))
    }

    pub fn log_state(&self, name: &str, dir: &str, particle_number: usize) -> std::io::Result<()> {
        let positions: Vec<Vec3> = unsafe {
            std::slice::from_raw_parts(
                self.buf_position.contents().as_ptr() as *const Vec3,
                particle_number,
            )
            .to_vec()
        };
        let directions: Vec<Vec3> = unsafe {
            std::slice::from_raw_parts(
                self.buf_direction.contents().as_ptr() as *const Vec3,
                particle_number,
            )
            .to_vec()
        };

        positions.write_npy(&format!("{}/{}_pos.npy", dir, name))?;
        directions.write_npy(&format!("{}/{}_dir.npy", dir, name))?;
        Ok(())
    }
}
