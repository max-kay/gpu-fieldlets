use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::*;
use std::ptr::NonNull;

use crate::build::SimulationParameters;
use crate::math::Vec3;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct GPUParams {
    pub particle_number: u32,
    pub rve_side_len: f32,
    pub epsilon_mat: f32,
    pub mag_dipole: f32,
    pub particle_vol: f32,
    pub e_sus_x: f32,
    pub e_sus_z: f32,
    pub radius_eq: f32,
    pub repulsion_factor: f32,
    pub t_drag: f32,
    pub r_drag: f32,
    pub ext_e_field: Vec3,
    pub ext_h_field: Vec3,
}

pub enum Stage {
    EField,
    HField,
    ElDipoles,
    PVels,
    UpdatePositions(f32),
    UpdateDirections(f32),
    Check,
}

pub struct MetalState {
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub pipeline_e_field: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub pipeline_h_field: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub pipeline_el_dipoles: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub pipeline_p_vels: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub pipeline_positions: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub pipeline_directions: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub pipeline_check: Retained<ProtocolObject<dyn MTLComputePipelineState>>,

    pub buf_positions: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub buf_directions: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub buf_el_dipole_moments: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub buf_e_field: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub buf_h_field: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub buf_pos_vel: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub buf_check_output: Retained<ProtocolObject<dyn MTLBuffer>>,
}

impl MetalState {
    pub fn new(params: &SimulationParameters, positions: &[Vec3], directions: &[Vec3]) -> Self {
        let device = MTLCreateSystemDefaultDevice().expect("no metal device");
        let queue = device.newCommandQueue().unwrap();

        let source = std::fs::read_to_string("src/lib.metal").unwrap();
        let source_ns = NSString::from_str(&source);
        let options = MTLCompileOptions::new();
        let library = device
            .newLibraryWithSource_options_error(&source_ns, Some(&options))
            .expect("failed to compile metal shader");

        let get_pipeline = |name: &str| {
            let name_ns = NSString::from_str(name);
            let function = library
                .newFunctionWithName(&name_ns)
                .expect("function not found");
            device
                .newComputePipelineStateWithFunction_error(&function)
                .unwrap()
        };

        let buf = |data: &[Vec3]| unsafe {
            device
                .newBufferWithBytes_length_options(
                    NonNull::new(data.as_ptr() as *mut _).unwrap(),
                    std::mem::size_of_val(data),
                    MTLResourceOptions::StorageModeShared,
                )
                .unwrap()
        };

        let zeros = vec![Vec3::default(); params.particle_number];
        let ext_e = vec![params.ext_e_field(0.0); params.particle_number];

        Self {
            queue,
            pipeline_e_field: get_pipeline("update_e_field"),
            pipeline_h_field: get_pipeline("update_h_field"),
            pipeline_el_dipoles: get_pipeline("update_el_dipoles"),
            pipeline_p_vels: get_pipeline("update_p_vels"),
            pipeline_positions: get_pipeline("update_positions"),
            pipeline_directions: get_pipeline("update_directions"),
            pipeline_check: get_pipeline("check_finite_and_max_vel"),

            buf_positions: buf(positions),
            buf_directions: buf(directions),
            buf_el_dipole_moments: buf(&zeros),
            buf_e_field: buf(&ext_e),
            buf_h_field: buf(&zeros),
            buf_pos_vel: buf(&zeros),
            buf_check_output: device
                .newBufferWithLength_options(8, MTLResourceOptions::StorageModeShared)
                .unwrap(),
        }
    }

    pub fn run_stage(&self, stage: Stage, params: &GPUParams) {
        let command_buffer = self.queue.commandBuffer().unwrap();
        let encoder = command_buffer.computeCommandEncoder().unwrap();

        let particle_number = params.particle_number as usize;

        unsafe {
            let (pipeline, buffers, delta_t) = match stage {
                Stage::EField => (
                    &*self.pipeline_e_field,
                    vec![
                        &*self.buf_positions,
                        &*self.buf_el_dipole_moments,
                        &*self.buf_e_field,
                    ],
                    None,
                ),
                Stage::HField => (
                    &*self.pipeline_h_field,
                    vec![
                        &*self.buf_positions,
                        &*self.buf_directions,
                        &*self.buf_h_field,
                    ],
                    None,
                ),
                Stage::ElDipoles => (
                    &*self.pipeline_el_dipoles,
                    vec![
                        &*self.buf_e_field,
                        &*self.buf_directions,
                        &*self.buf_el_dipole_moments,
                    ],
                    None,
                ),
                Stage::PVels => (
                    &*self.pipeline_p_vels,
                    vec![
                        &*self.buf_positions,
                        &*self.buf_directions,
                        &*self.buf_el_dipole_moments,
                        &*self.buf_pos_vel,
                    ],
                    None,
                ),
                Stage::UpdatePositions(dt) => (
                    &*self.pipeline_positions,
                    vec![&*self.buf_positions, &*self.buf_pos_vel],
                    Some(dt),
                ),
                Stage::UpdateDirections(dt) => (
                    &*self.pipeline_directions,
                    vec![
                        &*self.buf_directions,
                        &*self.buf_h_field,
                        &*self.buf_e_field,
                    ],
                    Some(dt),
                ),
                Stage::Check => {
                    encoder.setComputePipelineState(&self.pipeline_check);
                    let buffers = [
                        &*self.buf_el_dipole_moments,
                        &*self.buf_e_field,
                        &*self.buf_h_field,
                        &*self.buf_positions,
                        &*self.buf_directions,
                        &*self.buf_pos_vel,
                        &*self.buf_check_output,
                    ];
                    for (i, buf) in buffers.iter().enumerate() {
                        encoder.setBuffer_offset_atIndex(Some(*buf), 0, i as _);
                    }
                    encoder.setBytes_length_atIndex(
                        NonNull::new(params as *const _ as *mut _).unwrap(),
                        std::mem::size_of::<GPUParams>(),
                        buffers.len() as _,
                    );

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
                    return;
                }
            };

            encoder.setComputePipelineState(pipeline);
            for (i, buf) in buffers.iter().enumerate() {
                encoder.setBuffer_offset_atIndex(Some(*buf), 0, i as _);
            }

            let mut next_index = buffers.len();
            if let Some(dt) = delta_t {
                encoder.setBytes_length_atIndex(
                    NonNull::new(&dt as *const f32 as *mut _).unwrap(),
                    4,
                    next_index as _,
                );
                next_index += 1;
            }

            encoder.setBytes_length_atIndex(
                NonNull::new(params as *const _ as *mut _).unwrap(),
                std::mem::size_of::<GPUParams>(),
                next_index as _,
            );

            let grid_size = MTLSize {
                width: particle_number,
                height: 1,
                depth: 1,
            };
            let thread_group_size = MTLSize {
                width: pipeline
                    .maxTotalThreadsPerThreadgroup()
                    .min(particle_number),
                height: 1,
                depth: 1,
            };
            encoder.dispatchThreads_threadsPerThreadgroup(grid_size, thread_group_size);
        }

        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
    }
}
