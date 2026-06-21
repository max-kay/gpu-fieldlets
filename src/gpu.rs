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
#[derive(Copy, Clone, Debug)]
pub struct GpuParams {
    pub ext_h_field: Vec3,
    pub ext_e_field: Vec3,

    pub particle_number: u32,
    pub h_field_prefactor: f32,
    pub e_field_prefactor: f32,
    pub right_dipole_prefactor: f32,

    pub e_force_prefactor: f32,
    pub r_force_prefactor: f32,
    pub h_torque_prefactor: f32,
    pub e_torque_prefactor: f32,

    pub rve_side_len: f32,
    pub repulsion_factor: f32,
    pub radius_eq: f32,
    pub h_force_prefactor: f32,
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

pub struct GpuState {
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

impl GpuState {
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

impl GpuState {
    pub fn find_all_averages(&mut self, particle_number: usize) {
        macro_rules! get_avg {
            ($buf:expr) => {
                let slice = std::slice::from_raw_parts(
                    $buf.contents().as_ptr() as *const Vec3,
                    particle_number,
                );
                println!(
                    "average norm of {}: {}",
                    stringify!($buf),
                    slice.iter().map(|v| v.norm()).sum::<f32>() / particle_number as f32
                )
            };
        }
        unsafe {
            get_avg!(self.buf_position);
            get_avg!(self.buf_direction);
            get_avg!(self.buf_e_dipole);
            get_avg!(self.buf_e_field);
            get_avg!(self.buf_h_field);
            get_avg!(self.buf_velocity);
        }
    }
}

impl GpuState {
    pub fn run_stage(&self, stage: Stage, params: &GpuParams) {
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
                mem::size_of::<GpuParams>(),
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

    pub fn run_max_and_check(&self, params: &GpuParams) -> (f32, bool) {
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
                mem::size_of::<GpuParams>(),
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
