use std::{error::Error, fs::File, io::Write, path::Path};

use chrono::{self, Local};

mod build;
mod gpu;
mod math;
mod numpy;

use build::{SimulationBuilder, SimulationParameters};
use gpu::{GPUParams, MetalState};
use math::Vec3;
use numpy::Numpy;
use serde::Serialize;

use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
};

#[derive(Serialize)]
struct SimulationSummary {
    iterations_ran: usize,
    log_dir: String,
    success: bool,
    time_ran: f32,
}

impl SimulationSummary {
    pub fn to_json(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn Error>> {
        let file = File::create(path)?;
        serde_json::ser::to_writer_pretty(file, &self)?;
        Ok(())
    }
}

pub struct Simulation {
    params: SimulationParameters,
    metal: MetalState,
}

/// physics functions
impl Simulation {
    fn encode_e_field(
        &self,
        encoder: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>,
        time: f32,
    ) {
        let params = self.gpu_params(time, 0.0);
        self.metal.encode_dispatch(
            encoder,
            self.params.particle_number,
            &*self.metal.pipeline_e_field,
            &[
                &*self.metal.buf_positions,
                &*self.metal.buf_el_dipole_moments,
                &*self.metal.buf_e_field,
            ],
            &params,
        );
    }

    fn encode_h_field(
        &self,
        encoder: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>,
        time: f32,
    ) {
        let params = self.gpu_params(time, 0.0);
        self.metal.encode_dispatch(
            encoder,
            self.params.particle_number,
            &*self.metal.pipeline_h_field,
            &[
                &*self.metal.buf_positions,
                &*self.metal.buf_directions,
                &*self.metal.buf_h_field,
            ],
            &params,
        );
    }

    fn encode_el_dipoles(
        &self,
        encoder: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>,
    ) {
        let params = self.gpu_params(0.0, 0.0);
        self.metal.encode_dispatch(
            encoder,
            self.params.particle_number,
            &*self.metal.pipeline_el_dipoles,
            &[
                &*self.metal.buf_e_field,
                &*self.metal.buf_directions,
                &*self.metal.buf_el_dipole_moments,
            ],
            &params,
        );
    }

    fn encode_p_vels(
        &self,
        encoder: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>,
        time: f32,
    ) {
        let params = self.gpu_params(time, 0.0);
        self.metal.encode_dispatch(
            encoder,
            self.params.particle_number,
            &*self.metal.pipeline_p_vels,
            &[
                &*self.metal.buf_positions,
                &*self.metal.buf_directions,
                &*self.metal.buf_el_dipole_moments,
                &*self.metal.buf_pos_vel,
            ],
            &params,
        );
    }

    fn encode_d_vels(
        &self,
        encoder: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>,
        time: f32,
    ) {
        let params = self.gpu_params(time, 0.0);
        self.metal.encode_dispatch(
            encoder,
            self.params.particle_number,
            &*self.metal.pipeline_d_vels,
            &[
                &*self.metal.buf_h_field,
                &*self.metal.buf_e_field,
                &*self.metal.buf_directions,
                &*self.metal.buf_dir_vel,
            ],
            &params,
        );
    }

    fn encode_positions(
        &self,
        encoder: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>,
        delta_t: f32,
    ) {
        let params = self.gpu_params(0.0, delta_t);
        self.metal.encode_dispatch(
            encoder,
            self.params.particle_number,
            &*self.metal.pipeline_positions,
            &[&*self.metal.buf_positions, &*self.metal.buf_pos_vel],
            &params,
        );
    }

    fn encode_directions(
        &self,
        encoder: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>,
        delta_t: f32,
    ) {
        let params = self.gpu_params(0.0, delta_t);
        self.metal.encode_dispatch(
            encoder,
            self.params.particle_number,
            &*self.metal.pipeline_directions,
            &[&*self.metal.buf_directions, &*self.metal.buf_dir_vel],
            &params,
        );
    }

    fn gpu_params(&self, time: f32, delta_t: f32) -> GPUParams {
        GPUParams {
            particle_number: self.params.particle_number as u32,
            rve_side_len: self.params.rve_side_len,
            epsilon_mat: self.params.epsilon_mat,
            mag_dipole: self.params.mag_dipole,
            particle_vol: self.params.particle_vol,
            e_sus_x: self.params.e_sus_x,
            e_sus_z: self.params.e_sus_z,
            radius_eq: self.params.radius_eq,
            repulsion_factor: self.params.repulsion_factor,
            t_drag: self.params.t_drag(time),
            r_drag: self.params.r_drag(time),
            ext_e_field: self.params.ext_e_field(time),
            ext_h_field: self.params.ext_h_field(time),
            delta_t,
        }
    }

    pub fn update_el_dipoles(&self) {
        let command_buffer = self.metal.queue.commandBuffer().unwrap();
        let encoder = command_buffer.computeCommandEncoder().unwrap();
        self.encode_el_dipoles(&encoder);
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
    }
}

/// Method for the simulation
impl Simulation {
    fn new() -> SimulationBuilder {
        SimulationBuilder::default()
    }

    fn make_log_dir() -> String {
        let mut num = 0;
        loop {
            let dir = format!("out/{}_{}", Local::now().format("%Y-%m-%d_%H-%M-%S"), num);
            match std::fs::create_dir(&dir) {
                Ok(()) => return dir,
                Err(err) => match err.kind() {
                    std::io::ErrorKind::AlreadyExists => num += 1,
                    _ => panic!("could not create log dir"),
                },
            }
        }
    }

    fn run(&mut self) -> SimulationSummary {
        let log_dir = Self::make_log_dir();

        if let Err(err) = self.params.to_json(format!("{}/config.json", log_dir)) {
            eprintln!("could not log configuration: {err}")
        }

        let mut current_time = 0.0;
        let mut i = 0;
        let mut stdout = std::io::stdout();
        let mut delta_t;
        let mut last_delta_t = vec![0.0; 64];
        let mut log_step = 0;
        let start = std::time::Instant::now();
        let success = loop {
            let _ = write!(
                stdout.lock(),
                "{: >8.5} s/{: >6.2} s   ∆t = {: >8.2e} s  i = {: >8}\r",
                current_time,
                self.params.duration,
                last_delta_t.iter().sum::<f32>() / last_delta_t.len() as f32,
                i,
            );
            let _ = stdout.flush();

            if current_time > (log_step as f32 / self.params.log_frames as f32) {
                if let Err(err) = self.log_state(&format!("./{log_step:0>5}"), &log_dir) {
                    eprintln!("could not log: {err}");
                }
                log_step += 1;
            }

            // Batch 1: update fields and velocities, then check finiteness and max velocity
            let (largest_velocity, finite) = {
                let command_buffer = self.metal.queue.commandBuffer().unwrap();
                let encoder = command_buffer.computeCommandEncoder().unwrap();
                for _ in 0..2 {
                    self.encode_e_field(&encoder, current_time);
                    self.encode_el_dipoles(&encoder);
                }
                self.encode_h_field(&encoder, current_time);
                self.encode_p_vels(&encoder, current_time);
                self.encode_d_vels(&encoder, current_time);

                let params = self.gpu_params(current_time, 0.0);
                self.metal.encode_check(&encoder, &params);

                encoder.endEncoding();
                command_buffer.commit();
                command_buffer.waitUntilCompleted();

                let output = unsafe {
                    std::slice::from_raw_parts(
                        self.metal.buf_check_output.contents().as_ptr() as *const f32,
                        2,
                    )
                };
                (output[0], output[1] > 0.5)
            };

            if !finite {
                break false;
            }

            if largest_velocity > 0.0 {
                delta_t = (self.params.radius_eq * self.params.velocity_factor / largest_velocity)
                    .min(0.01);
            } else {
                delta_t = 0.001;
            }

            // Batch 2: update positions and directions
            {
                let command_buffer = self.metal.queue.commandBuffer().unwrap();
                let encoder = command_buffer.computeCommandEncoder().unwrap();
                self.encode_positions(&encoder, delta_t);
                self.encode_directions(&encoder, delta_t);
                encoder.endEncoding();
                command_buffer.commit();
                // We could skip waiting here, but logging might need it.
                command_buffer.waitUntilCompleted();
            }

            i += 1;
            current_time += delta_t;
            {
                let idx = i % last_delta_t.len();
                last_delta_t[idx] = delta_t;
            }
            if current_time > self.params.duration {
                break true;
            }
        };
        if let Err(err) = self.log_state(&format!("{log_step:0>5}"), &log_dir) {
            eprintln!("could not log: {err}");
        }
        let summary = SimulationSummary {
            iterations_ran: i,
            log_dir: log_dir.clone(),
            time_ran: current_time,
            success,
        };
        if let Err(err) = summary.to_json(format!("{}/summary.json", log_dir)) {
            eprintln!("could not log configuration: {err}")
        }
        println!("\nfinished in {:.0} s", start.elapsed().as_secs_f32());
        return summary;
    }

    fn log_state(&self, name: &str, dir: &str) -> std::io::Result<()> {
        let positions: Vec<Vec3> = unsafe {
            std::slice::from_raw_parts(
                self.metal.buf_positions.contents().as_ptr() as *const Vec3,
                self.params.particle_number,
            )
            .to_vec()
        };
        let directions: Vec<Vec3> = unsafe {
            std::slice::from_raw_parts(
                self.metal.buf_directions.contents().as_ptr() as *const Vec3,
                self.params.particle_number,
            )
            .to_vec()
        };

        positions.write_npy(&format!("{}/{}_pos.npy", dir, name))?;
        directions.write_npy(&format!("{}/{}_dir.npy", dir, name))?;
        Ok(())
    }
}

fn main() {
    let mut simulations: Vec<_> = vec![{
        let mut b = Simulation::new();
        b.duration = 0.3;
        b.particle_number = 2000;
        b.build()
    }];
    let len = simulations.len();
    for (i, s) in simulations.iter_mut().enumerate() {
        println!("\rsimulation of `{}` {}/{}", s.params.name, i + 1, len);
        let summary = s.run();
        if summary.success {
            println!("success");
        } else {
            println!("failed");
        }
        println!(
            "average ∆t = {:.3e} s",
            summary.time_ran / summary.iterations_ran as f32
        );
        println!()
    }
}
