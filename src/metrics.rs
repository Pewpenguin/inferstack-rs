use metrics::{counter, histogram, Counter, Histogram};
use std::sync::OnceLock;
use std::time::Instant;

pub struct Timer {
    start: Instant,
}

impl Timer {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    pub fn elapsed_seconds(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}

fn inference_requests_total() -> &'static Counter {
    static METRIC: OnceLock<Counter> = OnceLock::new();
    METRIC.get_or_init(|| counter!("inference_requests_total"))
}

fn inference_errors_total() -> &'static Counter {
    static METRIC: OnceLock<Counter> = OnceLock::new();
    METRIC.get_or_init(|| counter!("inference_errors_total"))
}

fn inference_duration_seconds() -> &'static Histogram {
    static METRIC: OnceLock<Histogram> = OnceLock::new();
    METRIC.get_or_init(|| histogram!("inference_duration_seconds"))
}

fn batch_inference_requests_total() -> &'static Counter {
    static METRIC: OnceLock<Counter> = OnceLock::new();
    METRIC.get_or_init(|| counter!("batch_inference_requests_total"))
}

fn cache_hits_total() -> &'static Counter {
    static METRIC: OnceLock<Counter> = OnceLock::new();
    METRIC.get_or_init(|| counter!("cache_hits_total"))
}

fn cache_misses_total() -> &'static Counter {
    static METRIC: OnceLock<Counter> = OnceLock::new();
    METRIC.get_or_init(|| counter!("cache_misses_total"))
}

pub fn record_inference_request() {
    inference_requests_total().increment(1);
}

pub fn record_inference_error() {
    inference_errors_total().increment(1);
}

pub fn record_inference_latency(duration: f64) {
    inference_duration_seconds().record(duration);
}

pub fn record_batch_request() {
    batch_inference_requests_total().increment(1);
}

pub fn record_cache_hit() {
    cache_hits_total().increment(1);
}

pub fn record_cache_miss() {
    cache_misses_total().increment(1);
}
