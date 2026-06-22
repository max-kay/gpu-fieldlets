use std::mem;
use std::path::Path;
use std::ptr::NonNull;

use dispatch2::DispatchData;
use image::{ImageBuffer, Rgba};
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
    CheckMaxVel,
    Position,
    Direction,
}

pub struct GpuState {
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    render_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
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
                render_queue: device.newCommandQueue().unwrap(),
                pipeline_e_field: get_pipeline("update_e_field"),
                pipeline_h_field: get_pipeline("update_h_field"),
                pipeline_e_dipole: get_pipeline("update_e_dipole"),
                pipeline_velocity: get_pipeline("update_velocity"),
                pipeline_position: get_pipeline("update_position"),
                pipeline_direction: get_pipeline("update_direction"),
                pipeline_check: get_pipeline("check_max_vel"),
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
                        MTLResourceOptions::StorageModePrivate,
                    )
                    .unwrap(),
                buf_e_field: device
                    .newBufferWithBytes_length_options(
                        NonNull::new(ext_e.as_ptr() as *mut _).unwrap(),
                        mem::size_of_val(&*ext_e),
                        MTLResourceOptions::StorageModePrivate,
                    )
                    .unwrap(),
                buf_h_field: device
                    .newBufferWithLength_options(
                        mem::size_of::<Vec3>() * params.particle_number,
                        MTLResourceOptions::StorageModePrivate,
                    )
                    .unwrap(),
                buf_velocity: device
                    .newBufferWithLength_options(
                        mem::size_of::<Vec3>() * params.particle_number,
                        MTLResourceOptions::StorageModePrivate,
                    )
                    .unwrap(),
                buf_check_output: device
                    .newBufferWithLength_options(
                        2 * mem::size_of::<f32>(),
                        MTLResourceOptions::StorageModeShared,
                    )
                    .unwrap(),
            }
        }
    }
}

impl GpuState {
    pub fn begin_pass(&self, params: &GpuParams, delta_t: Option<f32>) -> GpuPass<'_> {
        let command_buffer = self.queue.commandBuffer().unwrap();
        let encoder = command_buffer.computeCommandEncoder().unwrap();
        unsafe {
            encoder.setBytes_length_atIndex(
                NonNull::new(params as *const _ as *mut _).unwrap(),
                mem::size_of::<GpuParams>(),
                8,
            );
            if let Some(dt) = delta_t {
                encoder.setBytes_length_atIndex(
                    NonNull::new(&dt as *const f32 as *mut _).unwrap(),
                    mem::size_of::<f32>(),
                    9,
                );
            }
        }
        GpuPass {
            state: self,
            command_buffer,
            encoder,
            particle_number: params.particle_number as usize,
        }
    }

    pub fn read_max_vel(&self) -> f32 {
        unsafe { *(self.buf_check_output.contents().as_ptr() as *const f32) }
    }
}

pub struct GpuPass<'a> {
    state: &'a GpuState,
    command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    particle_number: usize,
}

impl<'a> GpuPass<'a> {
    pub fn dispatch(&self, stage: Stage) {
        let (pipeline, buffers) = match stage {
            Stage::EField => (
                &*self.state.pipeline_e_field,
                vec![
                    &*self.state.buf_position,
                    &*self.state.buf_e_dipole,
                    &*self.state.buf_e_field,
                ],
            ),
            Stage::HField => (
                &*self.state.pipeline_h_field,
                vec![
                    &*self.state.buf_position,
                    &*self.state.buf_direction,
                    &*self.state.buf_h_field,
                ],
            ),
            Stage::EDipole => (
                &*self.state.pipeline_e_dipole,
                vec![
                    &*self.state.buf_e_field,
                    &*self.state.buf_direction,
                    &*self.state.buf_e_dipole,
                ],
            ),
            Stage::Velocity => (
                &*self.state.pipeline_velocity,
                vec![
                    &*self.state.buf_position,
                    &*self.state.buf_direction,
                    &*self.state.buf_e_dipole,
                    &*self.state.buf_velocity,
                ],
            ),
            Stage::CheckMaxVel => (
                &*self.state.pipeline_check,
                vec![&*self.state.buf_velocity, &*self.state.buf_check_output],
            ),
            Stage::Position => (
                &*self.state.pipeline_position,
                vec![&*self.state.buf_position, &*self.state.buf_velocity],
            ),
            Stage::Direction => (
                &*self.state.pipeline_direction,
                vec![
                    &*self.state.buf_direction,
                    &*self.state.buf_h_field,
                    &*self.state.buf_e_field,
                ],
            ),
        };

        self.encoder.setComputePipelineState(pipeline);
        unsafe {
            for (i, buf) in buffers.iter().enumerate() {
                self.encoder.setBuffer_offset_atIndex(Some(*buf), 0, i as _);
            }
        }

        let (grid_size, thread_group_size) = if matches!(stage, Stage::CheckMaxVel) {
            (
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
            )
        } else {
            (
                MTLSize {
                    width: self.particle_number,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: pipeline
                        .maxTotalThreadsPerThreadgroup()
                        .min(self.particle_number),
                    height: 1,
                    depth: 1,
                },
            )
        };
        self.encoder
            .dispatchThreads_threadsPerThreadgroup(grid_size, thread_group_size);
    }

    pub fn commit(self) {
        self.encoder.endEncoding();
        self.command_buffer.commit();
    }

    pub fn commit_and_wait(self) {
        self.encoder.endEncoding();
        self.command_buffer.commit();
        self.command_buffer.waitUntilCompleted();
    }
}

impl GpuState {
    pub fn render_to_png(&self, spec: &FrameSpec, path: impl AsRef<Path>) {
        let copy_cb = self.queue.commandBuffer().unwrap();
        copy_cb.commit();
        copy_cb.waitUntilCompleted();

        let command_buffer = self.render_queue.commandBuffer().unwrap();
        let encoder = command_buffer.computeCommandEncoder().unwrap();
        encoder.setComputePipelineState(&self.pipeline_render);

        let device = self.queue.device();
        let buf_img = device
            .newBufferWithLength_options(
                (spec.dims[0] * spec.dims[1] as u32) as NSUInteger
                    * mem::size_of::<u32>() as NSUInteger,
                MTLResourceOptions::StorageModeShared,
            )
            .unwrap();

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&*buf_img), 0, 0);
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
        let path = path.as_ref().to_owned();
        let dims = spec.dims;
        let buffer = unsafe {
            std::slice::from_raw_parts(
                buf_img.contents().as_ptr() as *const u8,
                (dims[0] * dims[1]) as usize * mem::size_of::<u32>(),
            )
        };
        let buffer_vec = buffer.to_vec();
        std::thread::spawn(move || {
            let img_buffer = ImageBuffer::<Rgba<u8>, _>::from_raw(dims[0], dims[1], buffer_vec)
                .expect("Buffer size does not match the specified width and height");
            let _ = img_buffer.save(path);
        });
    }

    pub fn log_state(&self, prefix: String, particle_number: usize) -> std::io::Result<()> {
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

        positions.write_npy(&format!("{}_pos.npy", prefix))?;
        directions.write_npy(&format!("{}_dir.npy", prefix))?;
        Ok(())
    }
}
