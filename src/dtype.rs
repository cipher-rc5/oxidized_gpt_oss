// file: src/dtype.rs
// description: Low-precision quantization data types and packing helpers.
// author: cipher-rc5
// created: 2026-02-21
// modified: 2026-02-21
use bytemuck::{Pod, Zeroable};
use half::f16;
use std::fmt;

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub struct MxFp4(u8);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub struct F6E2M3(u8);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub struct F6E3M2(u8);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub struct F8E8M0(u8);

impl MxFp4 {
    // const SCALE_BITS: u8 = 8;

    pub fn new(value: u8) -> Self {
        Self(value & 0x0F)
    }

    pub fn from_f32(value: f32) -> Self {
        let abs_val = value.abs();
        let sign = if value < 0.0 { 0x08 } else { 0x00 };

        if abs_val == 0.0 {
            return Self(sign);
        }

        let exp = abs_val.log2().floor() as i32;
        let exp_clamped = exp.clamp(-2, 1);
        let exp_bits = ((exp_clamped + 2) as u8) & 0x03;

        let mantissa = (abs_val / 2.0_f32.powi(exp_clamped)) - 1.0;
        let mantissa_bit = if mantissa >= 0.5 { 0x01 } else { 0x00 };

        Self(sign | (exp_bits << 1) | mantissa_bit)
    }

    pub fn to_f32(self) -> f32 {
        let bits = self.0;
        let sign = if (bits & 0x08) != 0 { -1.0 } else { 1.0 };
        let exp_bits = (bits >> 1) & 0x03;
        let mantissa_bit = bits & 0x01;

        if exp_bits == 0 && mantissa_bit == 0 {
            return 0.0;
        }

        let exp = (exp_bits as i32) - 2;
        let mantissa = 1.0 + (mantissa_bit as f32) * 0.5;

        sign * mantissa * 2.0_f32.powi(exp)
    }

    pub fn to_f16(self) -> f16 {
        f16::from_f32(self.to_f32())
    }

    pub fn pack_pair(a: MxFp4, b: MxFp4) -> u8 {
        (a.0 & 0x0F) | ((b.0 & 0x0F) << 4)
    }

    pub fn unpack_pair(packed: u8) -> (MxFp4, MxFp4) {
        (MxFp4(packed & 0x0F), MxFp4((packed >> 4) & 0x0F))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MxBlock {
    pub scale: f16,
    pub values: [MxFp4; 32],
}

impl MxBlock {
    pub fn new(scale: f16) -> Self {
        Self {
            scale,
            values: [MxFp4(0); 32],
        }
    }

    pub fn from_f32_slice(values: &[f32]) -> Self {
        assert!(values.len() == 32, "MxBlock requires exactly 32 values");

        let max_abs = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

        let scale = if max_abs > 0.0 {
            f16::from_f32(max_abs / 7.5)
        } else {
            f16::from_f32(1.0)
        };

        let scale_f32 = scale.to_f32();
        let mut block = Self::new(scale);

        for (i, &value) in values.iter().enumerate() {
            let scaled = value / scale_f32;
            block.values[i] = MxFp4::from_f32(scaled);
        }

        block
    }

    pub fn to_f32_vec(&self) -> Vec<f32> {
        let scale = self.scale.to_f32();
        self.values.iter().map(|v| v.to_f32() * scale).collect()
    }

    pub fn pack(&self) -> Vec<u8> {
        let mut packed = Vec::with_capacity(18);

        packed.extend_from_slice(&self.scale.to_bits().to_le_bytes());

        for chunk in self.values.chunks(2) {
            let byte = if chunk.len() == 2 {
                MxFp4::pack_pair(chunk[0], chunk[1])
            } else {
                chunk[0].0
            };
            packed.push(byte);
        }

        packed
    }

    pub fn unpack(data: &[u8]) -> Option<Self> {
        if data.len() < 18 {
            return None;
        }

        let scale_bits = u16::from_le_bytes([data[0], data[1]]);
        let scale = f16::from_bits(scale_bits);

        let mut values = [MxFp4(0); 32];
        for (i, &byte) in data[2..18].iter().enumerate() {
            let (a, b) = MxFp4::unpack_pair(byte);
            values[i * 2] = a;
            if i * 2 + 1 < 32 {
                values[i * 2 + 1] = b;
            }
        }

        Some(Self { scale, values })
    }
}

impl F6E2M3 {
    pub fn from_f32(value: f32) -> Self {
        let abs_val = value.abs();
        let sign = if value < 0.0 { 0x20 } else { 0x00 };

        if abs_val == 0.0 {
            return Self(sign);
        }

        let exp = abs_val.log2().floor() as i32;
        let exp_clamped = exp.clamp(-1, 2);
        let exp_bits = ((exp_clamped + 1) as u8) & 0x03;

        let mantissa = (abs_val / 2.0_f32.powi(exp_clamped)) - 1.0;
        let mantissa_scaled = (mantissa * 7.0).round() as u8;
        let mantissa_bits = mantissa_scaled.min(7);

        Self(sign | (exp_bits << 3) | mantissa_bits)
    }

    pub fn to_f32(self) -> f32 {
        let bits = self.0;
        let sign = if (bits & 0x20) != 0 { -1.0 } else { 1.0 };
        let exp_bits = (bits >> 3) & 0x03;
        let mantissa_bits = bits & 0x07;

        if exp_bits == 0 && mantissa_bits == 0 {
            return 0.0;
        }

        let exp = (exp_bits as i32) - 1;
        let mantissa = 1.0 + (mantissa_bits as f32) / 7.0;

        sign * mantissa * 2.0_f32.powi(exp)
    }
}

impl F6E3M2 {
    pub fn from_f32(value: f32) -> Self {
        let abs_val = value.abs();
        let sign = if value < 0.0 { 0x20 } else { 0x00 };

        if abs_val == 0.0 {
            return Self(sign);
        }

        let exp = abs_val.log2().floor() as i32;
        let exp_clamped = exp.clamp(-3, 4);
        let exp_bits = ((exp_clamped + 3) as u8) & 0x07;

        let mantissa = (abs_val / 2.0_f32.powi(exp_clamped)) - 1.0;
        let mantissa_scaled = (mantissa * 3.0).round() as u8;
        let mantissa_bits = mantissa_scaled.min(3);

        Self(sign | (exp_bits << 2) | mantissa_bits)
    }

    pub fn to_f32(self) -> f32 {
        let bits = self.0;
        let sign = if (bits & 0x20) != 0 { -1.0 } else { 1.0 };
        let exp_bits = (bits >> 2) & 0x07;
        let mantissa_bits = bits & 0x03;

        if exp_bits == 0 && mantissa_bits == 0 {
            return 0.0;
        }

        let exp = (exp_bits as i32) - 3;
        let mantissa = 1.0 + (mantissa_bits as f32) / 3.0;

        sign * mantissa * 2.0_f32.powi(exp)
    }
}

impl F8E8M0 {
    pub fn from_f32(value: f32) -> Self {
        if value == 0.0 {
            return Self(0);
        }

        let exp = value.log2().floor() as i32;
        let exp_clamped = exp.clamp(-127, 127);
        let exp_biased = (exp_clamped + 127) as u8;

        Self(exp_biased)
    }

    pub fn to_f32(self) -> f32 {
        if self.0 == 0 {
            return 0.0;
        }

        let exp = (self.0 as i32) - 127;
        2.0_f32.powi(exp)
    }
}

impl fmt::Display for MxFp4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

impl fmt::Display for F6E2M3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

impl fmt::Display for F6E3M2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

impl fmt::Display for F8E8M0 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mxfp4_conversion() {
        let values = vec![0.0, 1.0, -1.0, 2.5, -3.5];
        for &val in &values {
            let fp4 = MxFp4::from_f32(val);
            let reconstructed = fp4.to_f32();
            assert!(
                (reconstructed - val).abs() < 1.0,
                "Value {} reconstructed as {}",
                val,
                reconstructed
            );
        }
    }

    #[test]
    fn test_mx_block() {
        let values: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let block = MxBlock::from_f32_slice(&values);
        let reconstructed = block.to_f32_vec();

        for (original, &recon) in values.iter().zip(reconstructed.iter()) {
            let error = (recon - original).abs() / original.max(1.0);
            assert!(error < 1.0, "Reconstruction error too high: {}", error);
        }
    }
}
