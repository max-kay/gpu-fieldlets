use super::{Float, PI};
use std::{
    fmt,
    ops::{Add, Div, Mul, Neg, Sub},
    ops::{AddAssign, Rem, RemAssign},
};

use rand::{
    distr::{OpenClosed01, Uniform},
    prelude::*,
};

use serde::{Serialize, Serializer};

pub trait GoodValues {
    fn is_finite(&self) -> bool;
}

impl GoodValues for Float {
    fn is_finite(&self) -> bool {
        Float::is_finite(*self)
    }
}

pub struct UnitVec;

impl Distribution<Vec3> for UnitVec {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec3 {
        fn box_muller<R: Rng + ?Sized>(rng: &mut R) -> (Float, Float) {
            let u1: Float = rng.sample(OpenClosed01);
            let u2: Float = rng.sample(OpenClosed01);
            let r = (-2.0 * (u1.ln())).sqrt();
            let theta = 2.0 * PI * u2;
            (r * theta.cos(), r * theta.sin())
        }
        let (x, y) = box_muller(rng);
        let (z, _) = box_muller(rng);
        let vec = Vec3 { x, y, z };
        vec.normalised()
    }
}

pub struct Bounded(pub Float);

impl Distribution<Vec3> for Bounded {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec3 {
        let distr = Uniform::new(-self.0 / 2.0, self.0 / 2.0).unwrap();
        Vec3 {
            x: rng.sample(distr),
            y: rng.sample(distr),
            z: rng.sample(distr),
        }
    }
}

#[derive(Clone, Copy, Default, PartialEq)]
#[repr(C)]
pub struct Vec3 {
    pub x: Float,
    pub y: Float,
    pub z: Float,
}

// To serialize the entire struct as a sequence
impl Serialize for Vec3 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeSeq;
        let mut seq = serializer.serialize_seq(Some(3))?;
        seq.serialize_element(&self.x)?;
        seq.serialize_element(&self.y)?;
        seq.serialize_element(&self.z)?;
        seq.end()
    }
}
impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}, {}]", self.x, self.y, self.z)
    }
}

impl fmt::Debug for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vec3")?;
        fmt::Display::fmt(self, f)
    }
}

impl GoodValues for Vec3 {
    fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

impl GoodValues for Vec<Vec3> {
    fn is_finite(&self) -> bool {
        self.iter()
            .map(|v| v.is_finite())
            .reduce(|acc, x| acc && x)
            .unwrap_or(true)
    }
}

impl Vec3 {
    pub fn new(x: Float, y: Float, z: Float) -> Self {
        Self { x, y, z }
    }

    pub fn dot(self, other: Self) -> Float {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn norm_sq(self) -> Float {
        self.dot(self)
    }

    pub fn norm(self) -> Float {
        self.norm_sq().sqrt()
    }

    pub fn normalised(self) -> Self {
        self / self.norm()
    }
}

impl Add for Vec3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}
impl Neg for Vec3 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}
impl Sub for Vec3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl Rem<Float> for Vec3 {
    type Output = Vec3;

    fn rem(self, rhs: Float) -> Self::Output {
        fn in_b(val: Float, side_len: Float) -> Float {
            val - (val / side_len).round() * side_len
        }
        Self {
            x: in_b(self.x, rhs),
            y: in_b(self.y, rhs),
            z: in_b(self.z, rhs),
        }
    }
}
impl RemAssign<Float> for Vec3 {
    fn rem_assign(&mut self, rhs: Float) {
        *self = *self % rhs;
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Mul<Float> for Vec3 {
    type Output = Self;

    fn mul(self, rhs: Float) -> Self::Output {
        Self {
            x: rhs * self.x,
            y: rhs * self.y,
            z: rhs * self.z,
        }
    }
}
impl Mul<Vec3> for Float {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Self::Output {
        Vec3 {
            x: self * rhs.x,
            y: self * rhs.y,
            z: self * rhs.z,
        }
    }
}

impl Div<Float> for Vec3 {
    type Output = Vec3;

    fn div(self, rhs: Float) -> Self::Output {
        self * (1.0 / rhs)
    }
}
