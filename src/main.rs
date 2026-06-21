use std::{error::Error, f32::consts::TAU, fs::File, io::Write, path::Path, process::Command};

use chrono::{self, Local};

mod gpu;
mod math;
mod numpy;
mod params;

use gpu::{MetalState, Stage};
use math::Vec3;
use params::{SimulationBuilder, SimulationParameters};
use serde::Serialize;

#[derive(Serialize)]
struct SimulationSummary {
    iterations_ran: usize,
    log_dir: String,
    success: bool,
    simulation_time: f64,
    real_time: f64,
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

        let mut current_time: f64 = 0.0;
        let mut i = 0;
        let mut stdout = std::io::stdout();
        let mut last_delta_t: Vec<f64> = vec![0.0; 64];
        let mut log_step = 0;
        let start = std::time::Instant::now();
        let success = loop {
            let _ = write!(
                stdout.lock(),
                "{: >8.5} s/{: >6.2} s   ∆t = {: >8.2e} s  i = {: >8}\r",
                current_time * self.params.char_time_scale,
                self.params.duration,
                last_delta_t.iter().sum::<f64>() / last_delta_t.len() as f64,
                i,
            );
            let _ = stdout.flush();

            if current_time
                >= (log_step as f64 / self.params.log_frames as f64) * self.params.duration
            {
                if let Err(err) = self.metal.run_plotting(
                    &format!("{log_step:0>5}"),
                    &log_dir,
                    &self.params.frame_spec(current_time),
                ) {
                    eprintln!("could not log: {err}");
                }
                log_step += 1;
            }

            let params = self.params.gpu_params(current_time);

            for _ in 0..3 {
                self.metal.run_stage(Stage::EField, &params);
                self.metal.run_stage(Stage::EDipole, &params);
            }
            self.metal.run_stage(Stage::HField, &params);

            if ("compare", i == 800).1 {
                dbg!(&params);
                self.metal.find_all_averages(&params);
                // FIXME: comparison
                self.metal.run_stage(Stage::Velocity, &params);
                let gpu_results: Vec<Vec3> = self.metal.get_velocity_buf(&params).into();
                self.metal.update_velocity(&params);
                let mut max_err = 0.0;
                let mut err_sum = 0.0;
                let mut max_angle = 0.0;
                let mut angle_sum = 0.0;
                for (rel_err, angle_err) in gpu_results
                    .iter()
                    .zip(self.metal.get_velocity_buf(&params))
                    .map(|(gpu, cpu)| {
                        let gpu_norm = gpu.norm();
                        let cpu_norm = cpu.norm();
                        (
                            (gpu_norm - cpu_norm) / cpu_norm,
                            (gpu.dot(*cpu) / (gpu_norm * cpu_norm)).acos(),
                        )
                    })
                {
                    err_sum += rel_err;
                    angle_sum += angle_err;
                    if max_err < rel_err {
                        max_err = rel_err
                    }
                    if max_angle < angle_err {
                        max_angle = angle_err
                    }
                }
                println!(
                    "Relative Magnitude error  max: {}  avg: {}",
                    max_err,
                    err_sum / params.particle_number as f32
                );
                println!(
                    "Angle error  max: {} degrees  avg: {} degrees",
                    max_angle / TAU * 360.0,
                    angle_sum / params.particle_number as f32 / TAU * 360.0
                );
                println!();
                // panic!("reached comparison");
            } else if ("run metal function", false).1 {
                self.metal.run_stage(Stage::Velocity, &params);
            } else {
                self.metal.update_velocity(&params);
            }
            let (max_vel, finite) = self.metal.run_max_and_check(&params);
            if !finite {
                break false;
            }
            let delta_t = (self.params.radius_eq * self.params.velocity_factor
                / (max_vel as f64 * self.params.rve_side_len))
                .min(2e-5);

            self.metal
                .run_stage(Stage::Position(delta_t as f32), &params);
            self.metal
                .run_stage(Stage::Direction(delta_t as f32), &params);

            i += 1;
            current_time += delta_t * self.params.char_time_scale;
            let idx = i % last_delta_t.len();
            last_delta_t[idx] = delta_t * self.params.char_time_scale;
            if current_time > self.params.duration {
                break true;
            }
        };

        if let Err(err) = self.metal.run_plotting(
            &format!("{log_step:0>5}"),
            &log_dir,
            &self.params.frame_spec(current_time),
        ) {
            eprintln!("could not log: {err}");
        }
        if let Err(err) = self.metal.log_state(
            &format!("{log_step:0>5}"),
            &log_dir,
            self.params.particle_number,
        ) {
            eprintln!("could not log: {err}");
        }

        let real_time = start.elapsed().as_secs_f64();
        let summary = SimulationSummary {
            iterations_ran: i,
            log_dir: log_dir.clone(),
            simulation_time: current_time,
            real_time,
            success,
        };
        if let Err(err) = summary.to_json(format!("{}/summary.json", log_dir)) {
            eprintln!("could not log configuration: {err}")
        }
        println!("\nfinished in {:.0} s", real_time);
        return summary;
    }
}

fn main() {
    #[allow(unused)]
    use params::ValueOrFn::{Fn, Value};

    let mut simulations: Vec<_> = vec![{
        let mut b = Simulation::new();
        b.duration = 0.08;
        b.particle_number = 100;
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
            summary.simulation_time / summary.iterations_ran as f64
        );
        println!("log dir: `{}`", summary.log_dir);
        println!();
        let anim_fname = format!("{}_{}.mp4", summary.log_dir, s.params.name);
        if Command::new("ffmpeg")
            .args(&["-loglevel", "error"])
            .arg("-hide_banner")
            .args(&["-framerate", "24"])
            .arg("-i")
            .arg(format!("{}/frame_%05d.png", summary.log_dir))
            .args(&["-c:v", "libx264"])
            .args(&["-pix_fmt", "yuv420p"])
            .args(&["-g", "12"])
            .arg(&anim_fname)
            .output()
            .is_ok()
        {
            println!("created animation")
        } else {
            println!("failed to create animation")
        };
        Command::new("open").arg(&anim_fname).output().unwrap();
    }
}
