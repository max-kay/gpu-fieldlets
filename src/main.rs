use std::{error::Error, fs::File, io::Write, path::Path, process::Command};

use chrono::{self, Local};

mod gpu;
mod math;
mod numpy;
mod params;

use gpu::{GPUParams, MetalState, Stage};
use math::Vec3;
use params::{SimulationBuilder, SimulationParameters};
use serde::Serialize;

#[derive(Serialize)]
struct SimulationSummary {
    iterations_ran: usize,
    log_dir: String,
    success: bool,
    simulation_time: f32,
    real_time: f32,
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

impl Simulation {}

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

            if current_time
                >= (log_step as f32 / self.params.log_frames as f32) * self.params.duration
            {
                self.metal.run_plotting(
                    &format!("{log_step:0>5}"),
                    &log_dir,
                    &self.params.frame_spec(current_time),
                );
                if let Err(err) = self.metal.log_state(
                    &format!("./{log_step:0>5}"),
                    &log_dir,
                    self.params.particle_number,
                ) {
                    eprintln!("could not log: {err}");
                }
                log_step += 1;
            }

            let params = self.params.gpu_params(current_time);

            for _ in 0..2 {
                self.metal.run_stage(Stage::EField, &params);
                self.metal.run_stage(Stage::HField, &params);
                self.metal.run_stage(Stage::ElDipoles, &params);
                self.metal.run_stage(Stage::HField, &params);
            }
            self.metal.run_stage(Stage::HField, &params);
            self.metal.run_stage(Stage::PVels, &params);

            let (max_vel, finite) = self.metal.run_max_and_check(&params);

            if !finite {
                break false;
            }

            let delta_t = (self.params.radius_eq * self.params.velocity_factor / max_vel).min(2e-5);

            self.metal
                .run_stage(Stage::UpdatePositions(delta_t), &params);
            self.metal
                .run_stage(Stage::UpdateDirections(delta_t), &params);

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
        if let Err(err) = self.metal.log_state(
            &format!("{log_step:0>5}"),
            &log_dir,
            self.params.particle_number,
        ) {
            eprintln!("could not log: {err}");
        }
        let real_time = start.elapsed().as_secs_f32();
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
    use params::ValueOrFn::{Fn, Value};
    let mut simulations: Vec<_> = vec![{
        let mut b = Simulation::new();
        b.duration = 0.2;
        b.particle_number = 200;
        b.h_field_norm = Value(0.0);
        // b.e_field_norm = Value(0.0);
        b.log_frames = 300;
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
            summary.simulation_time / summary.iterations_ran as f32
        );
        println!("log dir: `{}`", summary.log_dir);
        println!();
        if Command::new("ffmpeg")
            .args(&["-loglevel", "error"])
            .arg("-hide_banner")
            .args(&["-framerate", "24"])
            .arg("-i")
            .arg(format!("{}/frame_%05d.png", summary.log_dir))
            .args(&["-c:v", "libx264"])
            .args(&["-pix_fmt", "yuv420p"])
            .args(&["-g", "12"])
            .arg(format!("{}_{}.mp4", summary.log_dir, s.params.name))
            // ffmpeg -loglevel error -hide_banner -framerate 24 -i frame_%05d.png -c:v libx264 -pix_fmt yuv420p -g 12 output.mp4
            .output()
            .is_ok()
        {
            println!("created animation")
        } else {
            println!("failed to create animation")
        };
    }
}
