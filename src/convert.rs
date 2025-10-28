use crate::dtype::MxBlock;
use anyhow::{Context, Result};
use half::f16;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

pub struct ModelConverter;

impl ModelConverter {
    pub fn convert_f16_to_mxfp4(input_path: &Path, output_path: &Path) -> Result<()> {
        println!("Converting F16 weights to MXFP4 format...");

        let mut input = File::open(input_path).context("Failed to open input file")?;

        let mut buffer = Vec::new();
        input.read_to_end(&mut buffer)?;

        // Convert buffer to f16 values safely
        if buffer.len() % 2 != 0 {
            return Err(anyhow::anyhow!(
                "Buffer length must be even for f16 conversion"
            ));
        }

        let mut f16_values = Vec::with_capacity(buffer.len() / 2);
        for chunk in buffer.chunks_exact(2) {
            let bytes = [chunk[0], chunk[1]];
            f16_values.push(f16::from_le_bytes(bytes));
        }

        let num_blocks = (f16_values.len() + 31) / 32;
        let mut output_blocks = Vec::with_capacity(num_blocks);

        for chunk in f16_values.chunks(32) {
            let mut values = [0.0f32; 32];
            for (i, &v) in chunk.iter().enumerate() {
                values[i] = v.to_f32();
            }

            if chunk.len() < 32 {
                for i in chunk.len()..32 {
                    values[i] = 0.0;
                }
            }

            let block = MxBlock::from_f32_slice(&values);
            output_blocks.push(block);
        }

        let mut output = File::create(output_path).context("Failed to create output file")?;

        for block in &output_blocks {
            let packed = block.pack();
            output.write_all(&packed)?;
        }

        println!("Conversion complete!");
        println!("Original size: {} bytes", buffer.len());
        println!("Compressed size: {} bytes", output_blocks.len() * 18);
        println!(
            "Compression ratio: {:.2}x",
            buffer.len() as f32 / (output_blocks.len() * 18) as f32
        );

        Ok(())
    }

    pub fn validate_mxfp4_accuracy(original_path: &Path, converted_path: &Path) -> Result<()> {
        println!("Validating MXFP4 conversion accuracy...");

        let mut original = File::open(original_path)?;
        let mut original_buf = Vec::new();
        original.read_to_end(&mut original_buf)?;

        // Convert buffer to f16 values safely
        if original_buf.len() % 2 != 0 {
            return Err(anyhow::anyhow!(
                "Buffer length must be even for f16 conversion"
            ));
        }

        let mut f16_values = Vec::with_capacity(original_buf.len() / 2);
        for chunk in original_buf.chunks_exact(2) {
            let bytes = [chunk[0], chunk[1]];
            f16_values.push(f16::from_le_bytes(bytes));
        }

        let mut converted = File::open(converted_path)?;
        let mut converted_buf = Vec::new();
        converted.read_to_end(&mut converted_buf)?;

        let num_blocks = converted_buf.len() / 18;
        let mut reconstructed = Vec::with_capacity(f16_values.len());

        for i in 0..num_blocks {
            let block_data = &converted_buf[i * 18..(i + 1) * 18];
            let block = MxBlock::unpack(block_data).context("Failed to unpack block")?;
            let values = block.to_f32_vec();
            reconstructed.extend_from_slice(&values);
        }

        reconstructed.truncate(f16_values.len());

        let mut max_error = 0.0f32;
        let mut avg_error = 0.0f32;
        let mut relative_errors = Vec::new();

        for (&orig, &recon) in f16_values.iter().zip(reconstructed.iter()) {
            let orig_f32 = orig.to_f32();
            let abs_error = (orig_f32 - recon).abs();
            let rel_error = if orig_f32.abs() > 1e-6 {
                abs_error / orig_f32.abs()
            } else {
                abs_error
            };

            max_error = max_error.max(abs_error);
            avg_error += abs_error;
            relative_errors.push(rel_error);
        }

        avg_error /= f16_values.len() as f32;

        relative_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_rel_error = relative_errors[relative_errors.len() / 2];
        let p99_rel_error = relative_errors[(relative_errors.len() as f32 * 0.99) as usize];

        println!("\nAccuracy Analysis:");
        println!("  Max absolute error: {:.6}", max_error);
        println!("  Avg absolute error: {:.6}", avg_error);
        println!("  Median relative error: {:.4}%", median_rel_error * 100.0);
        println!(
            "  99th percentile relative error: {:.4}%",
            p99_rel_error * 100.0
        );

        if p99_rel_error > 0.20 {
            println!("\nWarning: High error rate detected. Model accuracy may be impacted.");
        } else {
            println!("\nValidation passed! Quantization quality is good.");
        }

        Ok(())
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use tempfile::NamedTempFile;

//     #[test]
//     fn test_conversion_roundtrip() {
//         let original_values: Vec<f16> = (0..1024)
//             .map(|i| f16::from_f32((i as f32) * 0.01))
//             .collect();

//         let original_bytes: &[u8] = bytemuck::cast_slice(&original_values);

//         let mut original_file = NamedTempFile::new().unwrap();
//         original_file.write_all(original_bytes).unwrap();

//         let converted_file = NamedTempFile::new().unwrap();

//         ModelConverter::convert_f16_to_mxfp4(original_file.path(), converted_file.path()).unwrap();

//         ModelConverter::validate_mxfp4_accuracy(original_file.path(), converted_file.path())
//             .unwrap();
//     }
// }
