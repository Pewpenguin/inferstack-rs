use std::sync::Arc;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tracing::{error, info};

use crate::config::AppConfig;
use crate::error::AppError;
use crate::metrics::{self, Timer};
use crate::model::ModelService;

#[derive(Debug, Deserialize)]
pub struct InferenceRequest {
    pub input: Vec<Vec<f32>>,
    #[serde(default)]
    batch: bool,
    #[serde(default)]
    pub model_version: Option<String>,
}

pub struct AppState {
    pub model_service: Arc<ModelService>,
    pub config: Arc<AppConfig>,
}

impl InferenceRequest {
    pub fn validate(
        &self,
        min_size: usize,
        max_size: usize,
        max_batch_size: usize,
    ) -> Result<(), AppError> {
        if self.input.is_empty() {
            metrics::record_validation_error("empty_batch");
            return Err(AppError::ValidationError(
                "Batch cannot be empty".to_string(),
            ));
        }

        if self.input.len() > max_batch_size {
            metrics::record_validation_error("batch_too_large");
            return Err(AppError::ValidationError(format!(
                "Batch size too large: {} (maximum: 32)",
                self.input.len()
            )));
        }

        for (i, input) in self.input.iter().enumerate() {
            let input_len = input.len();

            if input_len < min_size {
                metrics::record_validation_error("input_too_small");
                return Err(AppError::ValidationError(format!(
                    "Input at index {} size too small: {} (minimum: {})",
                    i, input_len, min_size
                )));
            }

            if input_len > max_size {
                metrics::record_validation_error("input_too_large");
                return Err(AppError::ValidationError(format!(
                    "Input at index {} size too large: {} (maximum: {})",
                    i, input_len, max_size
                )));
            }

            if input.iter().any(|&x| x.is_nan() || x.is_infinite()) {
                metrics::record_validation_error("invalid_values");
                return Err(AppError::ValidationError(format!(
                    "Input at index {} contains NaN or infinity values",
                    i
                )));
            }
        }

        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct InferenceResponse {
    output: Vec<Vec<f32>>,
}

pub async fn health_check() -> impl IntoResponse {
    info!("Health check requested");
    metrics::record_api_request("health", "success");
    StatusCode::OK
}

pub async fn inference_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<InferenceRequest>,
) -> impl IntoResponse {
    let model_service = &state.model_service;
    let config = &state.config;
    let batch_size = request.input.len();
    let request_id = headers
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let endpoint = if request.batch {
        "batch_inference"
    } else {
        "inference"
    };

    info!(
        endpoint,
        request_id = request_id.as_deref().unwrap_or("-"),
        batch_size,
        requested_model_version = request.model_version.as_deref().unwrap_or("auto"),
        "Inference request received"
    );

    let timer = Timer::new();
    metrics::record_api_request(endpoint, "started");

    if let Err(err) = request.validate(
        config.min_input_size,
        config.max_input_size,
        config.max_batch_size,
    ) {
        metrics::record_api_request(endpoint, "validation_error");
        error!(
            endpoint,
            request_id = request_id.as_deref().unwrap_or("-"),
            error = %err,
            "Request validation failed"
        );
        return err.into_response();
    }

    let result = if !request.batch {
        if batch_size != 1 {
            metrics::record_api_request(endpoint, "validation_error");
            let err =
                AppError::ValidationError("Non-batch mode requires exactly one input".to_string());
            error!(
                endpoint,
                request_id = request_id.as_deref().unwrap_or("-"),
                error = %err,
                "Request validation failed"
            );
            return err.into_response();
        }

        match model_service
            .infer_with_version_with_request_id(
                request.input[0].clone(),
                request.model_version.as_deref(),
                request_id.as_deref(),
            )
            .await
        {
            Ok((prediction, _executed_version)) => {
                metrics::record_api_request(endpoint, "success");
                let response = InferenceResponse {
                    output: vec![prediction],
                };
                (StatusCode::OK, Json(response)).into_response()
            }
            Err(e) => {
                metrics::record_api_request(endpoint, "error");
                metrics::record_error("api_inference");
                error!(
                    endpoint,
                    request_id = request_id.as_deref().unwrap_or("-"),
                    error = %e,
                    "Inference request failed"
                );
                e.into_response()
            }
        }
    } else {
        match process_batch(
            model_service,
            request.input,
            request.model_version.clone(),
            request_id.clone(),
        )
        .await
        {
            Ok(predictions) => {
                metrics::record_api_request(endpoint, "success");
                let response = InferenceResponse {
                    output: predictions,
                };
                (StatusCode::OK, Json(response)).into_response()
            }
            Err(e) => {
                metrics::record_api_request(endpoint, "error");
                metrics::record_error("api_batch_inference");
                error!(
                    endpoint,
                    request_id = request_id.as_deref().unwrap_or("-"),
                    error = %e,
                    "Batch inference request failed"
                );
                e.into_response()
            }
        }
    };

    metrics::record_api_latency(endpoint, timer.elapsed_seconds());

    result
}

async fn process_batch(
    model_service: &Arc<ModelService>,
    inputs: Vec<Vec<f32>>,
    model_version: Option<String>,
    request_id: Option<String>,
) -> Result<Vec<Vec<f32>>, AppError> {
    let batch_size = inputs.len();
    info!(
        request_id = request_id.as_deref().unwrap_or("-"),
        batch_size,
        requested_model_version = model_version.as_deref().unwrap_or("auto"),
        "Batch inference started"
    );
    metrics::record_batch_size(batch_size);

    let batch_timer = Timer::new();
    for _ in 0..batch_size {
        metrics::record_batch_item("started");
    }

    let outputs = model_service
        .infer_batch_with_version_with_request_id(
            inputs,
            model_version.as_deref(),
            request_id.as_deref(),
        )
        .await
        .map(|(values, _)| values);

    match &outputs {
        Ok(_) => {
            for _ in 0..batch_size {
                metrics::record_batch_item("success");
            }
        }
        Err(_) => {
            for _ in 0..batch_size {
                metrics::record_batch_item("error");
            }
        }
    }

    let duration = batch_timer.elapsed_seconds();
    info!(
        request_id = request_id.as_deref().unwrap_or("-"),
        batch_size,
        inference_duration_ms = (duration * 1000.0),
        "Batch inference completed"
    );
    metrics::record_api_latency("batch_processing", duration);
    metrics::record_batch_throughput(batch_size, duration);

    outputs
}

pub async fn batch_inference_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(mut request): Json<InferenceRequest>,
) -> impl IntoResponse {
    request.batch = true;
    inference_handler(State(state), headers, Json(request)).await
}

pub mod routes {
    use super::*;

    pub fn create_router(model_service: Arc<ModelService>, config: Arc<AppConfig>) -> Router {
        let state = Arc::new(AppState {
            model_service,
            config,
        });

        Router::new()
            .route("/health", get(health_check))
            .route("/infer", post(inference_handler))
            .route("/batch", post(batch_inference_handler))
            .with_state(state)
    }
}
