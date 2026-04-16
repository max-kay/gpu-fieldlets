use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use crate::math::Vec3;

fn format_shape(vec: &[usize]) -> String {
    let mut res = String::from("(");
    for val in vec.iter() {
        res.push_str(&val.to_string());
        res.push_str(", ");
    }
    res.pop();
    res.push(')');
    res
}

pub trait NumpyDType {
    const DTYPE: &'static str;
}

impl NumpyDType for f32 {
    const DTYPE: &'static str = "<f4";
}

impl NumpyDType for f64 {
    const DTYPE: &'static str = "<f8";
}

pub trait Numpy: Sized {
    type DTYPE: NumpyDType;
    fn shape(&self) -> Vec<usize>;
    fn write(self, writer: &mut impl Write) -> std::io::Result<()>;

    fn get_header(&self) -> Vec<u8> {
        let shape = format_shape(&self.shape());
        format!(
            "{{'descr': '{}', 'fortran_order': False, 'shape': {} }}",
            Self::DTYPE::DTYPE,
            shape
        )
        .into_bytes()
    }

    fn write_npy(self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let mut file = BufWriter::new(File::create(path)?);

        let mut header = self.get_header();

        let preamble_len = 10;
        let total_len = preamble_len + header.len();
        let padding_needed = (64 - (total_len % 64)) % 64 - 1;
        header.extend(std::iter::repeat(b'\x20').take(padding_needed));
        header.push(b'\n');
        let header_len = header.len() as u32;

        file.write_all(b"\x93NUMPY")?;
        file.write_all(&[3, 0])?;

        file.write_all(&header_len.to_le_bytes())?;
        file.write_all(&header)?;

        self.write(&mut file)?;
        file.flush()?;
        Ok(())
    }
}

impl Numpy for &[Vec3] {
    type DTYPE = f32;

    fn shape(&self) -> Vec<usize> {
        vec![self.len(), 3]
    }

    fn write(self, writer: &mut impl Write) -> std::io::Result<()> {
        for v in self {
            writer.write_all(&v.x.to_le_bytes())?;
            writer.write_all(&v.y.to_le_bytes())?;
            writer.write_all(&v.z.to_le_bytes())?;
        }
        Ok(())
    }
}

impl<const N: usize> Numpy for &[[Vec3; N]] {
    type DTYPE = f32;
    fn shape(&self) -> Vec<usize> {
        vec![self.len(), N, 3]
    }

    fn write(self, writer: &mut impl Write) -> std::io::Result<()> {
        for vs in self {
            for v in vs {
                writer.write_all(&v.x.to_le_bytes())?;
                writer.write_all(&v.y.to_le_bytes())?;
                writer.write_all(&v.z.to_le_bytes())?;
            }
        }
        Ok(())
    }
}
impl<const N: usize> Numpy for &[[f32; N]] {
    type DTYPE = f32;
    fn shape(&self) -> Vec<usize> {
        vec![self.len(), N]
    }

    fn write(self, writer: &mut impl Write) -> std::io::Result<()> {
        for vals in self {
            for val in vals {
                writer.write_all(&val.to_le_bytes())?;
            }
        }
        Ok(())
    }
}

impl Numpy for &[f32] {
    type DTYPE = f32;

    fn shape(&self) -> Vec<usize> {
        vec![self.len()]
    }

    fn write(self, writer: &mut impl Write) -> std::io::Result<()> {
        for val in self {
            writer.write_all(&val.to_le_bytes())?;
        }
        Ok(())
    }
}
