use std::{
    error::Error,
    fs::File,
    io::Write,
    path::Path,
    process::{Command, Stdio},
};

use chrono::{self, Local};

type Float = f64;
const PI: Float = std::f64::consts::PI as Float;
const EPSILON_0: Float = 8.8541878188e-12;
const MU_0: Float = 1.2566370612e-6;

mod build;
mod math;
mod numpy;

use build::{SimulationBuilder, SimulationParameters};
use math::{GoodValues, Vec3};
use numpy::Numpy;
use serde::Serialize;

#[derive(Serialize)]
struct SimulationSummary {
    iterations_ran: usize,
    log_dir: String,
    success: bool,
    time_ran: Float,
}

impl SimulationSummary {
    pub fn to_json(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn Error>> {
        let file = File::create(path)?;
        serde_json::ser::to_writer_pretty(file, &self)?;
        Ok(())
    }
}

struct Simulation {
    el_dipole_moments: Vec<Vec3>,
    e_field: Vec<Vec3>,
    h_field: Vec<Vec3>,
    positions: Vec<Vec3>,
    directions: Vec<Vec3>,
    pos_vel: Vec<Vec3>,
    dir_vel: Vec<Vec3>,
    params: SimulationParameters,
}

/// physics functions
impl Simulation {
    unsafe fn update_e_field(&mut self, time: Float) {
        unsafe {
            for i in 0..self.params.particle_number {
                let mut e_field_i = self.params.ext_e_field(time);
                for j in 0..self.params.particle_number {
                    if i == j {
                        continue;
                    }
                    let r_ji = (*self.positions.get_unchecked(i)
                        - *self.positions.get_unchecked(j))
                        % self.params.rve_side_len;
                    let dist = r_ji.norm();
                    let r_ji_hat = r_ji / dist;

                    let prefactor =
                        1.0 / (4.0 * PI * EPSILON_0 * self.params.epsilon_mat) / dist.powi(3);
                    let e_ji = prefactor
                        * (3.0
                            * (self.el_dipole_moments.get_unchecked(j).dot(r_ji_hat))
                            * r_ji_hat
                            - *self.el_dipole_moments.get_unchecked(j));
                    e_field_i += e_ji;
                }
                *self.e_field.get_unchecked_mut(i) = e_field_i;
            }
        }
    }

    unsafe fn update_h_field(&mut self, time: Float) {
        for i in 0..self.params.particle_number {
            unsafe {
                let mut h_field_i = self.params.ext_h_field(time);
                for j in 0..self.params.particle_number {
                    if i == j {
                        continue;
                    }
                    let r_ji = (*self.positions.get_unchecked(i)
                        - *self.positions.get_unchecked(j))
                        % self.params.rve_side_len;
                    let dist = r_ji.norm();
                    let r_ji_hat = r_ji / dist;

                    let prefactor = self.params.mag_dipole / (4.0 * PI) / dist.powi(3);
                    let h_ji = prefactor
                        * (3.0 * (self.directions.get_unchecked(j).dot(r_ji_hat)) * r_ji_hat
                            - *self.directions.get_unchecked(j));
                    h_field_i += h_ji;
                }
                *self.h_field.get_unchecked_mut(i) = h_field_i;
            }
        }
    }

    unsafe fn update_el_dipoles(&mut self) {
        for (i, p) in self.el_dipole_moments.iter_mut().enumerate() {
            unsafe {
                *p = self.params.particle_vol
                    * EPSILON_0
                    * (self.params.e_sus_x * *self.e_field.get_unchecked(i)
                        + (self.params.e_sus_z - self.params.e_sus_x)
                            * (self
                                .directions
                                .get_unchecked(i)
                                .dot(*self.e_field.get_unchecked(i)))
                            * *self.directions.get_unchecked(i))
            }
        }
    }

    unsafe fn update_p_vels(&mut self, time: Float) {
        self.pos_vel
            .iter_mut()
            .for_each(|p| *p = Vec3::new(0.0, 0.0, 0.0));
        for i in 0..self.params.particle_number {
            for j in (i + 1)..self.params.particle_number {
                unsafe {
                    let r_ji = (*self.positions.get_unchecked(i)
                        - *self.positions.get_unchecked(j))
                        % self.params.rve_side_len;
                    let dist = r_ji.norm();
                    let r_ji_hat = r_ji / dist;

                    // magnetic
                    let f_h1 = (self
                        .directions
                        .get_unchecked(i)
                        .dot(*self.directions.get_unchecked(j))
                        - 5.0
                            * (r_ji_hat.dot(*self.directions.get_unchecked(j)))
                            * (r_ji_hat.dot(*self.directions.get_unchecked(i))))
                        * r_ji_hat;
                    let f_h2 = r_ji_hat.dot(*self.directions.get_unchecked(i))
                        * *self.directions.get_unchecked(j)
                        + r_ji_hat.dot(*self.directions.get_unchecked(j))
                            * *self.directions.get_unchecked(i);
                    let f_h = 3.0 * MU_0 * self.params.mag_dipole.powi(2) / 4.0 / PI / dist.powi(4)
                        * (f_h1 + f_h2);

                    // electric
                    let f_e1 = (self
                        .el_dipole_moments
                        .get_unchecked(i)
                        .dot(*self.el_dipole_moments.get_unchecked(j))
                        - 5.0
                            * (r_ji_hat.dot(*self.el_dipole_moments.get_unchecked(j)))
                            * (r_ji_hat.dot(*self.el_dipole_moments.get_unchecked(i))))
                        * r_ji_hat;
                    let f_e2 = r_ji_hat.dot(*self.el_dipole_moments.get_unchecked(i))
                        * *self.el_dipole_moments.get_unchecked(j)
                        + r_ji_hat.dot(*self.el_dipole_moments.get_unchecked(j))
                            * *self.el_dipole_moments.get_unchecked(i);
                    let f_e = 3.0 / EPSILON_0 / self.params.epsilon_mat / 2.0 / PI / dist.powi(4)
                        * (f_e1 + f_e2);

                    // repulsive
                    let f_r = 3.0 * MU_0 * self.params.mag_dipole.powi(2)
                        / (2.0 * PI * (2.0 * self.params.radius_eq).powi(4))
                        * ((((-self.params.repulsion_factor
                            * (dist / (2.0 * self.params.radius_eq) - 1.0))
                            as f32)
                            .exp() as Float)
                            * r_ji_hat);

                    let f = (f_h + f_e + f_r) / self.params.t_drag(time);

                    *self.pos_vel.get_unchecked_mut(i) = *self.pos_vel.get_unchecked(i) + f;
                    *self.pos_vel.get_unchecked_mut(j) = *self.pos_vel.get_unchecked(j) - f;
                }
            }
        }
    }

    unsafe fn update_d_vels(&mut self, time: Float) {
        unsafe {
            for i in 0..self.params.particle_number {
                let magnetic = MU_0
                    * self.params.mag_dipole
                    * (*self.h_field.get_unchecked(i)
                        - *self.directions.get_unchecked(i)
                            * (self
                                .h_field
                                .get_unchecked(i)
                                .dot(*self.directions.get_unchecked(i))));
                let electric = self.params.particle_vol
                    * EPSILON_0
                    * (self.params.e_sus_z - self.params.e_sus_x)
                    * (self
                        .e_field
                        .get_unchecked(i)
                        .dot(*self.directions.get_unchecked(i)))
                    * (*self.e_field.get_unchecked(i)
                        - *self.directions.get_unchecked(i)
                            * (self
                                .e_field
                                .get_unchecked(i)
                                .dot(*self.directions.get_unchecked(i))));
                *self.dir_vel.get_unchecked_mut(i) =
                    (magnetic + electric) / self.params.r_drag(time);
            }
        }
    }

    unsafe fn update_positions(&mut self, delta_t: Float) {
        self.positions
            .iter_mut()
            .zip(self.pos_vel.iter())
            .for_each(|(p, v)| {
                *p = (*p + *v * delta_t) % self.params.rve_side_len;
            });
    }

    unsafe fn update_directions(&mut self, delta_t: Float) {
        self.directions
            .iter_mut()
            .zip(self.dir_vel.iter())
            .for_each(|(d, v)| {
                *d = (*d + delta_t * *v).normalised();
            });
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
        let mut delta_t = 0.0;
        let mut log_step = 0;
        let start = std::time::Instant::now();
        let success = loop {
            let _ = write!(
                stdout.lock(),
                "{: >8.5} s/{: >8.5} s   ∆t = {: >8.2e}   i = {: >8}\r",
                current_time,
                self.params.duration,
                delta_t,
                i
            );
            let _ = stdout.flush();

            if current_time > (log_step as Float / self.params.log_frames as Float) {
                if let Err(err) = self.log_state(&format!("./{log_step:0>5}"), &log_dir) {
                    eprintln!("could not log: {err}");
                }
                log_step += 1;
            }

            unsafe {
                for _ in 0..2 {
                    self.update_e_field(current_time);
                    self.update_el_dipoles();
                }

                self.update_h_field(current_time);

                self.update_p_vels(current_time);
                self.update_d_vels(current_time);

                let largest_velocity = self
                    .pos_vel
                    .iter()
                    .map(|v| v.norm())
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less))
                    .unwrap_or(0.0);
                delta_t = self.params.radius_eq / 3.0 / largest_velocity;

                self.update_positions(delta_t);
                self.update_directions(delta_t);
            }

            if !self.all_bufs_finite() {
                break false;
            }
            i += 1;
            current_time += delta_t;
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
        self.positions
            .write_npy(&format!("{}/{}_pos.npy", dir, name))?;
        self.directions
            .write_npy(&format!("{}/{}_dir.npy", dir, name))?;
        Ok(())
    }

    fn all_bufs_finite(&self) -> bool {
        self.el_dipole_moments.is_finite()
            && self.e_field.is_finite()
            && self.h_field.is_finite()
            && self.positions.is_finite()
            && self.directions.is_finite()
            && self.pos_vel.is_finite()
            && self.dir_vel.is_finite()
    }
}

fn start_plotting(path: &str) -> Result<std::process::Child, std::io::Error> {
    Command::new("./.venv/bin/python")
        .arg("./python/main.py")
        .arg(path)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
}

fn main() {
    let mut simulations: Vec<_> = (0..15)
        .map(|i| {
            let mut b = Simulation::new();
            b.repulsion_factor = i as Float * 2.0 + 14.0;
            b.build()
        })
        .collect();
    let len = simulations.len();

    let args: Vec<_> = std::env::args().collect();
    if args.len() == 2 && args[1] == "plot" {
        let mut children: Vec<(std::process::Child, String)> = Vec::new();
        for (i, s) in simulations.iter_mut().enumerate() {
            println!("\rsimulation of `{}` {}/{}", s.params.name, i + 1, len);
            let summary = s.run();
            match start_plotting(&summary.log_dir) {
                Ok(child) => children.push((child, s.params.name.clone())),
                Err(err) => eprintln!("could not launch plotting for `{}` {err}", s.params.name),
            }
        }

        for (mut child, name) in children {
            if let Err(err) = child.wait() {
                eprintln!("could not finish plotting for `{name}` {err}")
            }
        }
    } else {
        for (i, s) in simulations.iter_mut().enumerate() {
            println!("\rsimulation of `{}` {}/{}", s.params.name, i + 1, len);
            let summary = s.run();
            println!("beta = {}", s.params.repulsion_factor);
            if summary.success {
                println!("success");
            } else {
                println!("failed");
            }
            println!(
                "average ∆t = {:.3e} s",
                summary.time_ran / summary.iterations_ran as Float
            );
            println!()
        }
    }
}
