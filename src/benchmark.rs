// file: src/benchmark.rs
// description: Benchmark timing and throughput helpers for generation metrics.
// author: cipher-rc5

use std::time::{Duration, Instant};
use tracing::info;

#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub operation: String,
    pub duration: Duration,
    pub throughput: Option<f64>,
}

pub struct Benchmark {
    start: Instant,
    operation: String,
}

impl Benchmark {
    pub fn new(operation: impl Into<String>) -> Self {
        Self {
            start: Instant::now(),
            operation: operation.into(),
        }
    }

    pub fn finish(self) -> BenchmarkResult {
        let duration = self.start.elapsed();
        info!("{} took {:?}", self.operation, duration);

        BenchmarkResult {
            operation: self.operation,
            duration,
            throughput: None,
        }
    }

    pub fn finish_with_tokens(self, num_tokens: usize) -> BenchmarkResult {
        let duration = self.start.elapsed();
        let tokens_per_sec = num_tokens as f64 / duration.as_secs_f64();

        info!(
            "{} took {:?} ({:.2} tokens/sec)",
            self.operation, duration, tokens_per_sec
        );

        BenchmarkResult {
            operation: self.operation,
            duration,
            throughput: Some(tokens_per_sec),
        }
    }
}

pub struct PerformanceMetrics {
    pub tokens_generated: usize,
    pub total_time: Duration,
    pub time_to_first_token: Duration,
    pub tokens_per_second: f64,
    pub ms_per_token: f64,
}

impl PerformanceMetrics {
    pub fn new(
        tokens_generated: usize,
        total_time: Duration,
        time_to_first_token: Duration,
    ) -> Self {
        let tokens_per_second = tokens_generated as f64 / total_time.as_secs_f64();
        let ms_per_token = total_time.as_millis() as f64 / tokens_generated as f64;

        Self {
            tokens_generated,
            total_time,
            time_to_first_token,
            tokens_per_second,
            ms_per_token,
        }
    }

    pub fn print(&self) {
        info!("Performance Metrics:");
        info!("  Tokens generated: {}", self.tokens_generated);
        info!("  Total time: {:?}", self.total_time);
        info!("  Time to first token: {:?}", self.time_to_first_token);
        info!("  Throughput: {:.2} tokens/sec", self.tokens_per_second);
        info!("  Latency: {:.2} ms/token", self.ms_per_token);
    }
}
