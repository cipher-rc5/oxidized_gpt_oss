// file: src/mxfp4.rs
// description: MXFP4 E2M1 dequantization routines and tests for MoE checkpoint weights.
// author: cipher-rc5

use half::bf16;

const E2M1_MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

fn decode_e2m1(nibble: u8) -> f32 {
    let sign = (nibble & 0b1000) != 0;
    let magnitude = E2M1_MAGNITUDES[(nibble & 0b0111) as usize];
    if sign { -magnitude } else { magnitude }
}

pub fn dequantize_mxfp4(blocks: &[u8], scales: &[u16], rows: usize, cols: usize) -> Vec<u16> {
    assert!(cols % 32 == 0, "cols must be divisible by 32, got {cols}");

    let expected_blocks = rows * cols / 2;
    assert!(
        blocks.len() == expected_blocks,
        "blocks length mismatch: expected {}, got {}",
        expected_blocks,
        blocks.len()
    );

    let scales_per_row = cols / 32;
    let expected_scales = rows * scales_per_row;
    assert!(
        scales.len() == expected_scales,
        "scales length mismatch: expected {}, got {}",
        expected_scales,
        scales.len()
    );

    let mut out = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for col in 0..cols {
            let linear = row * cols + col;
            let packed = blocks[linear / 2];
            let nibble = if linear % 2 == 0 {
                packed & 0x0f
            } else {
                (packed >> 4) & 0x0f
            };

            let scale_idx = row * scales_per_row + (col / 32);
            let scale = bf16::from_bits(scales[scale_idx]).to_f32();
            let value = decode_e2m1(nibble) * scale;
            out.push(bf16::from_f32(value).to_bits());
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::dequantize_mxfp4;
    use half::bf16;

    const TABLE: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

    #[test]
    fn dequantizes_known_4x64_tensor() {
        let rows = 4;
        let cols = 64;

        let nibbles: Vec<u8> = (0..rows * cols).map(|i| (i % 16) as u8).collect();
        let blocks: Vec<u8> = nibbles
            .chunks_exact(2)
            .map(|p| (p[0] & 0x0f) | ((p[1] & 0x0f) << 4))
            .collect();

        let scales_f32 = [1.0_f32, 2.0, 0.5, 1.5, 0.25, 4.0, 3.0, 0.75];
        let scales: Vec<u16> = scales_f32
            .iter()
            .copied()
            .map(|v| bf16::from_f32(v).to_bits())
            .collect();

        let output = dequantize_mxfp4(&blocks, &scales, rows, cols);
        assert_eq!(output.len(), rows * cols);

        for row in 0..rows {
            for col in 0..cols {
                let idx = row * cols + col;
                let nibble = nibbles[idx];
                let sign = if (nibble & 0b1000) != 0 { -1.0 } else { 1.0 };
                let base = TABLE[(nibble & 0b0111) as usize];
                let scale = scales_f32[row * (cols / 32) + col / 32];
                let expected = bf16::from_f32(sign * base * scale).to_bits();
                assert_eq!(output[idx], expected, "mismatch at row={row}, col={col}");
            }
        }
    }
}
