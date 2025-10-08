use lazy_static::lazy_static;
use prometheus::{register_counter_vec, register_histogram_vec, CounterVec, HistogramVec};
use std::time::Instant;
use tracing::debug;
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
    
    pub static ref BATCH_COUNTER: CounterVec = register_counter_vec!(
        "inferstack_batch_items_total",
        "Total number of items processed in batches",
        &["status"]
    )
    .unwrap();
    
    pub static ref MODEL_VERSION_USAGE: CounterVec = register_counter_vec!(
        "inferstack_model_version_usage_total",
        "Total number of times each model version was used",
        &["version"]
    )
    .unwrap();
    
    pub static ref MODEL_VERSION_SUCCESS: CounterVec = register_counter_vec!(
        "inferstack_model_version_success_total",
        "Total number of successful inferences by model version",
        &["version"]
    )
    .unwrap();
    
    pub static ref MODEL_VERSION_ERROR: CounterVec = register_counter_vec!(
        "inferstack_model_version_error_total",
        "Total number of inference errors by model version",
        &["version"]
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
    
    pub static ref MODEL_VERSION_EXECUTION_TIME: HistogramVec = register_histogram_vec!(
        "inferstack_model_version_execution_seconds",
        "Model execution time in seconds by version",
        &["version"],
        vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
    )
    .unwrap();
    
    pub static ref FALLBACK_COUNTER: CounterVec = register_counter_vec!(
        "inferstack_fallback_total",
        "Total number of fallbacks to default model version",
        &["from_version", "to_version", "reason"]
    )
    .unwrap();
    
    pub static ref INPUT_VALIDATION_COUNTER: CounterVec = register_counter_vec!(
        "inferstack_input_validation_total",
        "Total number of input validation operations",
        &["result", "reason"]
    )
    .unwrap();
    
    pub static ref INPUT_PREPROCESSING_COUNTER: CounterVec = register_counter_vec!(
        "inferstack_input_preprocessing_total",
        "Total number of input preprocessing operations",
        &["operation"]
    )
    .unwrap();
    
    pub static ref SYSTEM_HEALTH: CounterVec = register_counter_vec!(
        "inferstack_system_health_total",
        "System health indicators",
        &["component", "status"]
    )
    .unwrap();

    pub static ref BATCH_SIZE: HistogramVec = register_histogram_vec!(
        "inferstack_batch_size",
        "Size of input batches",
        &[],
        vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0]
    )
    .unwrap();
    
    pub static ref BATCH_THROUGHPUT: HistogramVec = register_histogram_vec!(
        "inferstack_batch_throughput_items_per_second",
        "Batch processing throughput in items per second",
        &[],
        vec![1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0]
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

pub fn record_model_execution_time_with_version(duration: f64, version: &str) {
    record_model_execution_time(duration);
    MODEL_VERSION_EXECUTION_TIME.with_label_values(&[version]).observe(duration);
}

pub fn record_model_version_usage(version: &str) {
    MODEL_VERSION_USAGE.with_label_values(&[version]).inc();
}

pub fn record_model_version_success(version: &str) {
    MODEL_VERSION_SUCCESS.with_label_values(&[version]).inc();
}

pub fn record_model_version_error(version: &str) {
    MODEL_VERSION_ERROR.with_label_values(&[version]).inc();
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

pub fn record_batch_item(status: &str) {
    BATCH_COUNTER.with_label_values::<&str>(&[status]).inc();
}

pub fn record_batch_throughput(batch_size: usize, duration_seconds: f64) {
    if duration_seconds > 0.0 {
        let items_per_second = batch_size as f64 / duration_seconds;
        debug!("Batch throughput: {:.2} items/second", items_per_second);
        BATCH_THROUGHPUT.with_label_values::<&str>(&[]).observe(items_per_second);
    }
}