use std::{
    fs::File,
    io::{BufWriter, Write as _},
    panic,
    path::Path,
};

type Float = f64;
const NUMPY_FLOAT_DESCR: &str = "<f8";
const PI: Float = 3.14159265;
const EPSILON_0: Float = 8.8541878188e-12;
const MU_0: Float = 1.2566370612e-6;

mod build;
mod math;

use build::WorldBuilder;
use math::{GoodValues, Vec3};

struct World {
    positions: Vec<Vec3>,
    directions: Vec<Vec3>,
    el_dipole_moments: Vec<Vec3>,
    e_field: Vec<Vec3>,
    h_field: Vec<Vec3>,

    param: WorldBuilder,

    particle_vol: Float,
    rve_side_len: Float,
    radius_eq: Float,
    t_drag: Float,
    r_drag: Float,
    e_sus_x: Float,
    e_sus_z: Float,
    mag_dipole: Float,

    log_dir: String,
}

impl World {
    pub fn new() -> WorldBuilder {
        WorldBuilder::default()
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

/// physics functions
impl World {
    fn calc_e_field(&mut self) {
        for i in 0..self.param.particle_number {
            let mut e_field_i = self.param.e_field;
            for j in 0..self.param.particle_number {
                if i == j {
                    continue;
                }
                let r_ji = (self.positions[i] - self.positions[j]) % self.rve_side_len;
                let dist = r_ji.norm();
                let r_ji_hat = r_ji / dist;

                let prefactor =
                    1.0 / (4.0 * PI * EPSILON_0 * self.param.epsilon_mat) / dist.powi(3);
                let e_ji = prefactor
                    * (3.0 * (self.el_dipole_moments[j].dot(r_ji_hat)) * r_ji_hat
                        - self.el_dipole_moments[j]);
                e_field_i += e_ji;
            }
            self.e_field[i] = e_field_i;
        }
    }

    fn calc_h_field(&mut self) {
        for i in 0..self.param.particle_number {
            let mut h_field_i = self.param.h_field;
            for j in 0..self.param.particle_number {
                if i == j {
                    continue;
                }
                let r_ji = (self.positions[i] - self.positions[j]) % self.rve_side_len;
                let dist = r_ji.norm();
                let r_ji_hat = r_ji / dist;

                let prefactor = self.mag_dipole / (4.0 * PI) / dist.powi(3);
                let e_ji = prefactor
                    * (3.0 * (self.directions[j].dot(r_ji_hat)) * r_ji_hat - self.directions[j]);
                h_field_i += e_ji;
            }
            self.h_field[i] = h_field_i;
        }
    }

    fn calc_dipoles(&mut self) {
        for _ in 0..10 {
            // on the first run the calulation of the e_field is approximated by using the old
            // dipole moments
            self.calc_e_field();
            for (i, p) in self.el_dipole_moments.iter_mut().enumerate() {
                *p = self.particle_vol
                    * EPSILON_0
                    * (self.e_sus_x * self.e_field[i]
                        + (self.e_sus_z - self.e_sus_x)
                            * (self.directions[i].dot(self.e_field[i]))
                            * self.directions[i])
            }
        }
    }

    fn calc_p_vels(&self, buf: &mut [Vec3]) {
        buf.iter_mut().for_each(|p| *p = Vec3::new(0.0, 0.0, 0.0));
        for i in 0..self.param.particle_number {
            for j in (i + 1)..self.param.particle_number {
                let r_ji = (self.positions[i] - self.positions[j]) % self.rve_side_len;
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
                let f_h =
                    3.0 * MU_0 * self.mag_dipole.powi(2) / 4.0 / PI / dist.powi(4) * (f_h1 + f_h2);

                // electric
                let f_e1 = (self.el_dipole_moments[i].dot(self.el_dipole_moments[j])
                    - 5.0
                        * (r_ji_hat.dot(self.el_dipole_moments[j]))
                        * (r_ji_hat.dot(self.el_dipole_moments[i])))
                    * r_ji_hat;
                let f_e2 = r_ji_hat.dot(self.el_dipole_moments[i]) * self.el_dipole_moments[j]
                    + r_ji_hat.dot(self.el_dipole_moments[j]) * self.el_dipole_moments[i];
                let f_e = 3.0 / EPSILON_0 / self.param.epsilon_mat / 2.0 / PI / dist.powi(4)
                    * (f_e1 + f_e2);

                // repulsive
                let f_r = MU_0 * self.mag_dipole.powi(2) * self.param.h_field.norm_sq() / 4.0 / PI
                    * ((-self.param.repulsion_factor * (dist / (2.0 * self.radius_eq - 1.0)))
                        .exp()
                        * r_ji_hat);

                let f = f_r + f_h + f_e;

                buf[i] += f / self.t_drag;
                buf[j] += -f / self.t_drag;
            }
        }
    }

    fn calc_d_vels(&self, buf: &mut [Vec3]) {
        for i in 0..self.param.particle_number {
            let magnetic = MU_0
                * self.mag_dipole
                * (self.h_field[i]
                    - self.directions[i] * (self.h_field[i].dot(self.directions[i])));
            let electric = self.particle_vol
                * EPSILON_0
                * (self.e_sus_z - self.e_sus_x)
                * (self.e_field[i].dot(self.directions[i]))
                * (self.e_field[i]
                    - self.directions[i] * (self.e_field[i].dot(self.directions[i])));
            buf[i] = (magnetic + electric) / self.r_drag;
        }
    }
}

/// Method for the simulation
impl World {
    fn step_by(&mut self, p_buf: &mut [Vec3], d_buf: &mut [Vec3]) {
        self.calc_dipoles();
        self.calc_h_field();
        self.calc_e_field();
        self.calc_p_vels(p_buf);
        self.calc_d_vels(d_buf);
        self.positions
            .iter_mut()
            .zip(p_buf.iter())
            .for_each(|(p, v)| {
                *p = (*p + self.param.delta_time * *v) % self.rve_side_len;
            });
        self.directions
            .iter_mut()
            .zip(d_buf.iter())
            .for_each(|(d, v)| {
                *d = (*d + self.param.delta_time * *v).normalised();
            });

        if self.check_all_bufs("end of step") {
            panic!("Cannot continue with invalid buffers!")
        }
    }

    pub fn run(&mut self) {
        let mut p_buf = vec![Vec3::default(); self.positions.len()];
        let mut d_buf = vec![Vec3::default(); self.positions.len()];
        let iterations = (self.param.duration / self.param.delta_time) as usize;
        for i in 0..iterations {
            println!("{}/{}", i + 1, iterations);
            if i % self.param.log_step == 0 {
                if let Err(err) = self.debug_file(&format!("{i:0>8}")) {
                    eprintln!("could not log: {err}");
                }
            }
            self.step_by(&mut p_buf, &mut d_buf);
        }
        if let Err(err) = self.debug_file(&format!("{iterations:0>8}")) {
            eprintln!("could not log: {err}");
        }
    }

    pub fn debug_file(&self, name: &str) -> std::io::Result<()> {
        write_array(
            &self.positions,
            &format!("{}/{}_pos.npy", self.log_dir, name),
        )?;
        write_array(
            &self.directions,
            &format!("{}/{}_dir.npy", self.log_dir, name),
        )?;
        Ok(())
    }
}

pub fn write_array(data: &[Vec3], path: impl AsRef<Path>) -> std::io::Result<()> {
    let mut file = BufWriter::new(File::create(path)?);
    let shape = format!("({}, 3)", data.len());
    let header = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {} }}",
        NUMPY_FLOAT_DESCR, shape
    );
    let mut header_bytes = header.into_bytes();
    let preamble_len = 10;
    let total_len = preamble_len + header_bytes.len();
    let padding_needed = 64 - total_len % 64;

    header_bytes.extend(std::iter::repeat(b'\x20').take(padding_needed));

    file.write_all(b"\x93NUMPY")?;
    file.write_all(&[1, 0])?; // Version 1.0
    let h_len = header_bytes.len() as u16;
    file.write_all(&h_len.to_le_bytes())?;
    file.write_all(&header_bytes)?;

    // Safety: Vec3 is repr(C) and contains three f32s (12 bytes total)
    // Writing the entire buffer at once is much faster
    let data_ptr = data.as_ptr() as *const u8;
    let data_len = data.len() * std::mem::size_of::<Vec3>();
    unsafe {
        let bytes = std::slice::from_raw_parts(data_ptr, data_len);
        file.write_all(bytes)?;
    }

    file.flush()?;
    Ok(())
}

fn main() {
    let mut world = World::new().build();
    world.run();
}
