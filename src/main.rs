use std::{error::Error, fs::File, io::Write, path::Path, process::Command};

use chrono::{self, Local};

mod gpu;
mod math;
mod numpy;
mod params;

use gpu::{GpuState, Stage};
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
    gpu_state: GpuState,
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
                current_time,
                self.params.duration,
                last_delta_t.iter().sum::<f64>() / last_delta_t.len() as f64,
                i,
            );
            let _ = stdout.flush();

            if current_time
                >= (log_step as f64 / self.params.log_frames as f64) * self.params.duration
            {
                if let Err(err) = self.gpu_state.run_plotting(
                    &format!("{log_step:0>5}"),
                    &log_dir,
                    &self.params.frame_spec(current_time),
                ) {
                    eprintln!("could not log: {err}");
                }
                log_step += 1;
            }

            let params = self.params.gpu_params(current_time);

            if false {
                println!();
                dbg!(&params);
                println!(
                    "scaled delta t: {}",
                    last_delta_t[(i - 1) % last_delta_t.len()] / self.params.char_time_scale
                );
                self.gpu_state
                    .find_all_averages(self.params.particle_number);
            }

            for _ in 0..3 {
                self.gpu_state.run_stage(Stage::EField, &params);
                self.gpu_state.run_stage(Stage::EDipole, &params);
            }
            self.gpu_state.run_stage(Stage::HField, &params);

            self.gpu_state.run_stage(Stage::Velocity, &params);
            let (max_vel, finite) = self.gpu_state.run_max_and_check(&params);
            if !finite {
                break false;
            }
            let delta_t = (self.params.radius_eq * self.params.velocity_factor
                / (max_vel as f64 * self.params.rve_side_len))
                .min(2e-5);

            self.gpu_state
                .run_stage(Stage::Position(delta_t as f32), &params);
            self.gpu_state
                .run_stage(Stage::Direction(delta_t as f32), &params);

            i += 1;
            current_time += delta_t * self.params.char_time_scale;
            let idx = i % last_delta_t.len();
            last_delta_t[idx] = delta_t * self.params.char_time_scale;
            if current_time > self.params.duration {
                break true;
            }
        };

        if let Err(err) = self.gpu_state.run_plotting(
            &format!("{log_step:0>5}"),
            &log_dir,
            &self.params.frame_spec(current_time),
        ) {
            eprintln!("could not log: {err}");
        }
        if let Err(err) = self.gpu_state.log_state(
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
        b.h_field_norm = Value(0.0);
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
