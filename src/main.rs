use std::{
    error::Error,
    fs::{self, File},
    io::Write,
    path::Path,
    process::Command,
};

use chrono::{self, Local};

mod gpu;
mod math;
mod numpy;
mod params;

use gpu::{GpuState, Stage};
use math::Vec3;
use params::{SimulationBuilder, SimulationParameters};

#[derive(serde::Serialize)]
pub struct SimulationSummary {
    iterations_ran: usize,
    log_dir: String,
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
    fn run(&mut self, log_dir: &str) -> SimulationSummary {
        if let Err(err) = self
            .params
            .to_json(format!("{}/calculated_params.json", log_dir))
        {
            eprintln!("could not log configuration: {err}")
        }

        let mut current_time: f64 = 0.0;
        let mut i = 0;
        let mut stdout = std::io::stdout();
        let mut last_delta_t: Vec<f64> = vec![0.0; 64];
        let mut log_step = 0;
        let start = std::time::Instant::now();
        let mut params;
        loop {
            let _ = write!(
                stdout.lock(),
                "{: >8.5} s/{: >6.2} s   ∆t = {: >8.2e} s  i = {: >8}\r",
                current_time,
                self.params.duration,
                last_delta_t.iter().sum::<f64>() / last_delta_t.len() as f64,
                i,
            );
            let _ = stdout.flush();

            params = self.params.gpu_params(current_time);

            let pass1 = self
                .gpu_state
                .begin_pass(&params, None, "Update Fields and Velocity");
            for _ in 0..2 {
                pass1.dispatch(Stage::EField);
                pass1.dispatch(Stage::EDipole);
            }
            pass1.dispatch(Stage::HField);
            pass1.dispatch(Stage::Velocity);
            pass1.dispatch(Stage::MaxVel);
            pass1.commit_and_wait();

            let max_vel = self.gpu_state.read_max_vel();
            let delta_t = (self.params.radius_eq * self.params.velocity_factor
                / (max_vel as f64 * self.params.rve_side_len))
                .min(2e-5 / self.params.char_time_scale);

            let pass2 = self.gpu_state.begin_pass(
                &params,
                Some(delta_t as f32),
                "Update Position and Direction",
            );
            pass2.dispatch(Stage::Position);
            pass2.dispatch(Stage::Direction);
            pass2.commit();

            if current_time
                >= (log_step as f64 / self.params.log_frames as f64) * self.params.duration
            {
                self.gpu_state.render_to_png(
                    &self.params.frame_spec(current_time),
                    &format!("{}/frame_{:0>5}.png", log_dir, log_step),
                    format!("{: >8.5} s", current_time),
                    params.get_h_text(),
                    params.get_e_text(),
                );
                log_step += 1;
            }

            i += 1;
            current_time += delta_t * self.params.char_time_scale;
            let idx = i % last_delta_t.len();
            last_delta_t[idx] = delta_t * self.params.char_time_scale;
            if current_time > self.params.duration {
                break;
            }
        }

        self.gpu_state.render_to_png(
            &self.params.frame_spec(current_time),
            &format!("{}/frame_{:0>5}.png", log_dir, log_step),
            format!("{: >8.5} s", current_time),
            params.get_h_text(),
            params.get_e_text(),
        );
        if let Err(err) = self.gpu_state.log_state(
            format!("{}/end_state", log_dir),
            self.params.particle_number,
        ) {
            eprintln!("could not log: {err}");
        }

        let real_time = start.elapsed().as_secs_f64();
        let summary = SimulationSummary {
            iterations_ran: i,
            log_dir: log_dir.to_string(),
            simulation_time: current_time,
            real_time,
        };
        if let Err(err) = summary.to_json(format!("{}/summary.json", log_dir)) {
            eprintln!("could not log configuration: {err}")
        }
        println!();
        return summary;
    }
}

fn format_seconds(seconds: f64) -> String {
    let hours = (seconds / (60.0 * 60.0)).floor() as u32;
    let minutes = ((seconds - hours as f64 * (60.0 * 60.0)) / 60.0).floor() as u32;
    if hours != 0 {
        format!(
            "{}h {}min {}s",
            hours,
            minutes,
            (seconds - hours as f64 * (60.0 * 60.0) - minutes as f64 * 60.0).floor() as u32
        )
    } else if minutes != 0 {
        format!(
            "{}min {:.3}s",
            minutes,
            (seconds - hours as f64 * (60.0 * 60.0) - minutes as f64 * 60.0)
        )
    } else {
        format!("{:.5}s", seconds)
    }
}
fn make_log_dir() -> String {
    let mut num = 0;
    loop {
        let dir = format!("out/{}_{}", Local::now().format("%Y-%m-%d_%H-%M-%S"), num);
        match std::fs::create_dir_all(&dir) {
            Ok(()) => return dir,
            Err(err) => match err.kind() {
                std::io::ErrorKind::AlreadyExists => num += 1,
                _ => panic!("could not create log dir"),
            },
        }
    }
}

fn main() {
    let path = Path::new("configs");

    let mut simulations: Vec<SimulationBuilder> = Vec::new();

    for entry in fs::read_dir(path).expect("could not find path").flatten() {
        let file_path = entry.path();
        if !(file_path.is_file() && file_path.extension().and_then(|s| s.to_str()) == Some("json"))
        {
            continue;
        }
        if let Ok(content) = fs::read_to_string(&file_path) {
            match serde_json::from_str::<SimulationBuilder>(&content) {
                Ok(mut b) => {
                    b.duration = 0.001;
                    b.particle_number = 1000;
                    b.log_frames = 500;
                    simulations.push(b)
                }
                Err(e) => eprintln!("Failed to read {:?}: {}", file_path, e),
            }
        }
    }
    let len = simulations.len();
    for (i, s) in simulations.into_iter().enumerate() {
        let log_dir = make_log_dir();
        println!("{}/{} simulation of `{}`", i + 1, len, &s.name);
        println!("output directory: {log_dir}");
        let name = s.name.clone();
        let summary = s.run(&log_dir);
        println!(
            "finished in {} average ∆t = {:.3e} s",
            format_seconds(summary.real_time),
            summary.simulation_time / summary.iterations_ran as f64,
        );
        let anim_fname = format!("{}_{}.mp4", log_dir, &name);
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
            println!("created animation: {anim_fname}")
        } else {
            println!("failed to create animation")
        };
        // Command::new("open").arg(&anim_fname).output().unwrap();
        println!();
    }
}
