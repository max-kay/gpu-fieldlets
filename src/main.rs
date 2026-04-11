use std::{
    io::Write,
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

use build::{SimulationBuilder, SimulationParameters, ValueOrFn};
use math::{GoodValues, Vec3};
use numpy::Numpy;

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
    fn update_e_field(&mut self, time: Float) {
        for i in 0..self.params.particle_number {
            let mut e_field_i = self.params.ext_e_field(time);
            for j in 0..self.params.particle_number {
                if i == j {
                    continue;
                }
                let r_ji = (self.positions[i] - self.positions[j]) % self.params.rve_side_len;
                let dist = r_ji.norm();
                let r_ji_hat = r_ji / dist;

                let prefactor =
                    1.0 / (4.0 * PI * EPSILON_0 * self.params.epsilon_mat) / dist.powi(3);
                let e_ji = prefactor
                    * (3.0 * (self.el_dipole_moments[j].dot(r_ji_hat)) * r_ji_hat
                        - self.el_dipole_moments[j]);
                e_field_i += e_ji;
            }
            self.e_field[i] = e_field_i;
        }
    }

    fn update_h_field(&mut self, time: Float) {
        for i in 0..self.params.particle_number {
            let mut h_field_i = self.params.ext_h_field(time);
            for j in 0..self.params.particle_number {
                if i == j {
                    continue;
                }
                let r_ji = (self.positions[i] - self.positions[j]) % self.params.rve_side_len;
                let dist = r_ji.norm();
                let r_ji_hat = r_ji / dist;

                let prefactor = self.params.mag_dipole / (4.0 * PI) / dist.powi(3);
                let h_ji = prefactor
                    * (3.0 * (self.directions[j].dot(r_ji_hat)) * r_ji_hat - self.directions[j]);
                h_field_i += h_ji;
            }
            self.h_field[i] = h_field_i;
        }
    }

    fn update_el_dipoles(&mut self) {
        for (i, p) in self.el_dipole_moments.iter_mut().enumerate() {
            *p = self.params.particle_vol
                * EPSILON_0
                * (self.params.e_sus_x * self.e_field[i]
                    + (self.params.e_sus_z - self.params.e_sus_x)
                        * (self.directions[i].dot(self.e_field[i]))
                        * self.directions[i])
        }
    }

    fn update_p_vels(&mut self, time: Float) {
        self.pos_vel
            .iter_mut()
            .for_each(|p| *p = Vec3::new(0.0, 0.0, 0.0));
        for i in 0..self.params.particle_number {
            for j in (i + 1)..self.params.particle_number {
                let r_ji = (self.positions[i] - self.positions[j]) % self.params.rve_side_len;
                let dist = r_ji.norm();
                let r_ji_hat = r_ji / dist;

                // magnetic
                let f_h1 = (self.directions[i].dot(self.directions[j])
                    - 5.0
                        * (r_ji_hat.dot(self.directions[j]))
                        * (r_ji_hat.dot(self.directions[i])))
                    * r_ji_hat;
                let f_h2 = r_ji_hat.dot(self.directions[i]) * self.directions[j]
                    + r_ji_hat.dot(self.directions[j]) * self.directions[i];
                let f_h = 3.0 * MU_0 * self.params.mag_dipole.powi(2) / 4.0 / PI / dist.powi(4)
                    * (f_h1 + f_h2);

                // electric
                let f_e1 = (self.el_dipole_moments[i].dot(self.el_dipole_moments[j])
                    - 5.0
                        * (r_ji_hat.dot(self.el_dipole_moments[j]))
                        * (r_ji_hat.dot(self.el_dipole_moments[i])))
                    * r_ji_hat;
                let f_e2 = r_ji_hat.dot(self.el_dipole_moments[i]) * self.el_dipole_moments[j]
                    + r_ji_hat.dot(self.el_dipole_moments[j]) * self.el_dipole_moments[i];
                let f_e = 3.0 / EPSILON_0 / self.params.epsilon_mat / 2.0 / PI / dist.powi(4)
                    * (f_e1 + f_e2);

                // repulsive
                let f_r = if dist < self.params.radius_eq * 6.0 {
                    3.0 * MU_0 * self.params.mag_dipole.powi(2)
                        / (2.0 * PI * (2.0 * self.params.radius_eq).powi(4))
                        * ((((-self.params.repulsion_factor
                            * (dist / (2.0 * self.params.radius_eq) - 1.0))
                            as f32)
                            .exp() as Float)
                            * r_ji_hat)
                } else {
                    Vec3::new(0.0, 0.0, 0.0)
                };

                let f = (f_h + f_e + f_r) / self.params.t_drag(time);

                self.pos_vel[i] = self.pos_vel[i] + f;
                self.pos_vel[j] = self.pos_vel[j] - f;
            }
        }
    }

    fn update_d_vels(&mut self, time: Float) {
        for i in 0..self.params.particle_number {
            let magnetic = MU_0
                * self.params.mag_dipole
                * (self.h_field[i]
                    - self.directions[i] * (self.h_field[i].dot(self.directions[i])));
            let electric = self.params.particle_vol
                * EPSILON_0
                * (self.params.e_sus_z - self.params.e_sus_x)
                * (self.e_field[i].dot(self.directions[i]))
                * (self.e_field[i]
                    - self.directions[i] * (self.e_field[i].dot(self.directions[i])));
            self.dir_vel[i] = (magnetic + electric) / self.params.r_drag(time);
        }
    }

    fn update_positions(&mut self) {
        self.positions
            .iter_mut()
            .zip(self.pos_vel.iter())
            .for_each(|(p, v)| {
                let mut translation: Vec3 = *v * self.params.delta_time;
                if translation.norm() > self.params.radius_eq / 3.0 {
                    translation = translation.normalised() * self.params.radius_eq / 3.0;
                }
                *p = (*p + translation) % self.params.rve_side_len;
            });
    }

    fn update_directions(&mut self) {
        self.directions
            .iter_mut()
            .zip(self.dir_vel.iter())
            .for_each(|(d, v)| {
                *d = (*d + self.params.delta_time * *v).normalised();
            });
    }
}

/// Method for the simulation
impl Simulation {
    fn new() -> SimulationBuilder {
        SimulationBuilder::default()
    }

    fn run(&mut self) -> String {
        let log_dir = format!("out/{}", Local::now().format("%Y-%m-%d_%H-%M-%S"));
        if let Err(err) = std::fs::create_dir_all(&log_dir) {
            eprintln!("could not make log dir: {err}")
        }

        if let Err(err) = self.params.to_json(format!("{}/config.json", log_dir)) {
            eprintln!("could not log configuration: {err}")
        }

        let mut current_time = 0.0;
        let mut i = 0;
        let mut stdout = std::io::stdout();
        loop {
            i += 1;
            current_time += self.params.delta_time;
            if current_time > self.params.duration {
                break;
            }

            let _ = write!(
                stdout.lock(),
                "\r{: >8.5} s/{: >8.5} s",
                current_time,
                self.params.duration
            );
            let _ = stdout.flush();
            if i % self.params.log_step == 0 {
                if let Err(err) = self.log_state(&format!("./{i:0>8}"), &log_dir) {
                    eprintln!("could not log: {err}");
                }
            }

            for _ in 0..2 {
                self.update_e_field(current_time);
                self.update_el_dipoles();
            }

            self.update_h_field(current_time);

            self.update_p_vels(current_time);
            self.update_d_vels(current_time);

            self.update_positions();
            self.update_directions();

            if self.check_all_bufs("") {
                eprintln!("Cannot continue with invalid buffers!");
                break;
            }
        }
        println!();
        if let Err(err) = self.log_state(&format!("{i:0>8}"), &log_dir) {
            eprintln!("could not log: {err}");
        }
        log_dir
    }

    fn log_state(&self, name: &str, dir: &str) -> std::io::Result<()> {
        self.positions
            .write_npy(&format!("{}/{}_pos.npy", dir, name))?;
        self.directions
            .write_npy(&format!("{}/{}_dir.npy", dir, name))?;
        Ok(())
    }

    fn check_all_bufs(&self, prepend: &str) -> bool {
        let mut failed = false;
        if !self.positions.is_finite() {
            failed = true;
            eprintln!("{prepend} positions was not finite");
        };
        if !self.directions.is_finite() {
            failed = true;
            eprintln!("{prepend} directions was not finite");
        };
        if !self.e_field.is_finite() {
            failed = true;
            eprintln!("{prepend} e_field was not finite");
        };
        if !self.h_field.is_finite() {
            failed = true;
            eprintln!("{prepend} h_field was not finite");
        };
        if !self.el_dipole_moments.is_finite() {
            failed = true;
            eprintln!("{prepend} e_dipole_moments was not finite");
        };
        failed
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
    let mut children = Vec::new();
    let mut simulations = vec![
        {
            // fn dir(t: Float) -> Vec3 {
            //     let theta = 3.0 * t * 2.0 * PI;
            //     Vec3::new(theta.sin(), theta.cos(), 0.0)
            // }
            let mut builder = Simulation::new();
            // builder.e_field_dir = Vec3::new(1.0, 0.0, 0.0).into();
            // builder.h_field_dir = ValueOrFn::Fn(dir);
            // builder.name = "rotating H field".into();
            builder.build()
        },
        // {
        //     fn dir(t: Float) -> Vec3 {
        //         let theta = 3.0 * t * 2.0 * PI;
        //         Vec3::new(theta.sin(), theta.cos(), 0.0)
        //     }
        //     let mut builder = Simulation::new();
        //     builder.h_field_dir = Vec3::new(1.0, 0.0, 0.0).into();
        //     builder.e_field_dir = ValueOrFn::Fn(dir);
        //     builder.name = "rotating E field".into();
        //     builder.build()
        // },
    ];

    let len = simulations.len();
    for (i, s) in simulations.iter_mut().enumerate() {
        println!("simulation of `{}` {}/{}", s.params.name, i + 1, len);
        let log_dir = s.run();
        match start_plotting(&log_dir) {
            Ok(child) => children.push((child, s.params.name.clone())),
            Err(err) => eprintln!("could not launch plotting for `{}` {err}", s.params.name),
        }
    }

    for (mut child, name) in children {
        if let Err(err) = child.wait() {
            eprintln!("could not finish plotting for `{name}` {err}")
        }
    }
}
