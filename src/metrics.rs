use lazy_static::lazy_static;
use prometheus::{register_counter_vec, register_histogram_vec, CounterVec, HistogramVec};
use std::time::Instant;

lazy_static! {
    pub static ref INFERENCE_COUNTER: CounterVec = register_counter_vec!(
        "inferstack_inference_total",
        "Total number of inference requests",
        &["status"]
    )
    .unwrap();

    pub static ref API_REQUESTS: CounterVec = register_counter_vec!(
        "inferstack_api_requests_total",
        "Total number of API requests",
        &["endpoint", "status"]
    )
    .unwrap();

    pub static ref CACHE_COUNTER: CounterVec = register_counter_vec!(
        "inferstack_cache_operations_total",
        "Total number of cache operations",
        &["operation", "result"]
    )
    .unwrap();
    
    pub static ref VALIDATION_COUNTER: CounterVec = register_counter_vec!(
        "inferstack_validation_total",
        "Total number of validation errors",
        &["type"]
    )
    .unwrap();
    
    pub static ref RATE_LIMIT_COUNTER: CounterVec = register_counter_vec!(
        "inferstack_rate_limit_total",
        "Total number of rate limit events",
        &["action"]
    )
    .unwrap();

    pub static ref INFERENCE_LATENCY: HistogramVec = register_histogram_vec!(
        "inferstack_inference_duration_seconds",
        "Inference request duration in seconds",
        &["cached"],
        vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )
    .unwrap();

    pub static ref API_LATENCY: HistogramVec = register_histogram_vec!(
        "inferstack_api_latency_seconds",
        "API request latency in seconds",
        &["endpoint"],
        vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )
    .unwrap();

    pub static ref MODEL_EXECUTION_TIME: HistogramVec = register_histogram_vec!(
        "inferstack_model_execution_seconds",
        "Model execution time in seconds",
        &[],
        vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
    )
    .unwrap();

    pub static ref BATCH_SIZE: HistogramVec = register_histogram_vec!(
        "inferstack_batch_size",
        "Size of input batches",
        &[],
        vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0]
    )
    .unwrap();

    pub static ref ERROR_COUNTER: CounterVec = register_counter_vec!(
        "inferstack_errors_total",
        "Total number of errors",
        &["type"]
    )
    .unwrap();
}

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

pub fn record_inference_request(status: &str) {
    INFERENCE_COUNTER.with_label_values(&[status]).inc();
}

pub fn record_api_request(endpoint: &str, status: &str) {
    API_REQUESTS.with_label_values(&[endpoint, status]).inc();
}

pub fn record_cache_operation(operation: &str, result: &str) {
    CACHE_COUNTER.with_label_values(&[operation, result]).inc();
}

pub fn record_inference_latency(duration: f64, cached: bool) {
    let cached_str = if cached { "true" } else { "false" };
    INFERENCE_LATENCY.with_label_values(&[cached_str]).observe(duration);
}

pub fn record_api_latency(endpoint: &str, duration: f64) {
    API_LATENCY.with_label_values::<&str>(&[endpoint]).observe(duration);
}

pub fn record_model_execution_time(duration: f64) {
    MODEL_EXECUTION_TIME.with_label_values::<&str>(&[]).observe(duration);
}

pub fn record_batch_size(size: usize) {
    BATCH_SIZE.with_label_values::<&str>(&[]).observe(size as f64);
}

pub fn record_error(error_type: &str) {
    ERROR_COUNTER.with_label_values::<&str>(&[error_type]).inc();
}

pub fn record_validation_error(validation_type: &str) {
    VALIDATION_COUNTER.with_label_values::<&str>(&[validation_type]).inc();
    record_error("validation");
}

pub fn record_rate_limit(action: &str) {
    RATE_LIMIT_COUNTER.with_label_values::<&str>(&[action]).inc();
}