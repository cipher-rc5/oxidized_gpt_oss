// use anyhow::Result;
// use gpt_metal_mxfp4::benchmark::{print_system_info, Benchmark};

// fn main() -> Result<()> {
//     print_system_info()?;

//     println!("Initializing Metal backend...");
//     let bench = Benchmark::new()?;

//     let results = bench.run_full_benchmark_suite()?;

//     println!("\n=== Summary ===\n");

//     let mut total_gflops = 0.0;
//     let mut count_gflops = 0;

//     for result in &results {
//         if let Some(gflops) = result.throughput_gflops {
//             total_gflops += gflops;
//             count_gflops += 1;
//         }
//     }

//     if count_gflops > 0 {
//         println!(
//             "Average GEMM Performance: {:.2} GFLOPS",
//             total_gflops / count_gflops as f64
//         );
//     }

//     println!("\nBenchmark complete!");

//     Ok(())
// }
