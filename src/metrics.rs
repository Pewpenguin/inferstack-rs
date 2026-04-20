use metrics::{counter, histogram};
use std::time::Instant;
use tracing::debug;

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

pub fn record_inference_request(status: &str) {
    counter!("inferstack_inference_total", "status" => status.to_string()).increment(1);
}

pub fn record_api_request(endpoint: &str, status: &str) {
    counter!(
        "inferstack_api_requests_total",
        "endpoint" => endpoint.to_string(),
        "status" => status.to_string()
    )
    .increment(1);
}

pub fn record_cache_operation(operation: &str, result: &str) {
    counter!(
        "inferstack_cache_operations_total",
        "operation" => operation.to_string(),
        "result" => result.to_string()
    )
    .increment(1);
}

pub fn record_inference_latency(duration: f64, cached: bool, version: &str) {
    let cached_str = if cached { "true" } else { "false" };
    histogram!(
        "inferstack_inference_duration_seconds",
        "version" => version.to_string(),
        "cached" => cached_str.to_string()
    )
    .record(duration);
}

pub fn record_api_latency(endpoint: &str, duration: f64) {
    histogram!("inferstack_api_latency_seconds", "endpoint" => endpoint.to_string())
        .record(duration);
}

pub fn record_model_execution_time(duration: f64) {
    histogram!("inferstack_model_execution_seconds").record(duration);
}

pub fn record_model_execution_time_with_version(duration: f64, version: &str) {
    record_model_execution_time(duration);
    histogram!(
        "inferstack_model_version_execution_seconds",
        "version" => version.to_string()
    )
    .record(duration);
}

pub fn record_model_version_usage(version: &str) {
    counter!(
        "inferstack_model_version_usage_total",
        "version" => version.to_string()
    )
    .increment(1);
}

pub fn record_model_version_success(version: &str) {
    counter!(
        "inferstack_model_version_success_total",
        "version" => version.to_string()
    )
    .increment(1);
}

pub fn record_model_version_error(version: &str) {
    counter!(
        "inferstack_model_version_error_total",
        "version" => version.to_string()
    )
    .increment(1);
}

pub fn record_batch_size(size: usize) {
    histogram!("inferstack_batch_size").record(size as f64);
}

pub fn record_error(error_type: &str) {
    counter!("inferstack_errors_total", "type" => error_type.to_string()).increment(1);
}

pub fn record_validation_error(validation_type: &str) {
    counter!(
        "inferstack_validation_total",
        "type" => validation_type.to_string()
    )
    .increment(1);
    record_error("validation");
}

pub fn record_rate_limit(action: &str) {
    counter!("inferstack_rate_limit_total", "action" => action.to_string()).increment(1);
}

pub fn record_batch_item(status: &str) {
    counter!("inferstack_batch_items_total", "status" => status.to_string()).increment(1);
}

pub fn record_batch_throughput(batch_size: usize, duration_seconds: f64) {
    if duration_seconds > 0.0 {
        let items_per_second = batch_size as f64 / duration_seconds;
        debug!("Batch throughput: {:.2} items/second", items_per_second);
        histogram!("inferstack_batch_throughput_items_per_second").record(items_per_second);
    }
}
