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
            return Err(AppError::ValidationError(
                "Batch cannot be empty".to_string(),
            ));
        }

        if self.input.len() > max_batch_size {
            return Err(AppError::ValidationError(format!(
                "Batch size too large: {} (maximum: 32)",
                self.input.len()
            )));
        }

        for (i, input) in self.input.iter().enumerate() {
            let input_len = input.len();

            if input_len < min_size {
                return Err(AppError::ValidationError(format!(
                    "Input at index {} size too small: {} (minimum: {})",
                    i, input_len, min_size
                )));
            }

            if input_len > max_size {
                return Err(AppError::ValidationError(format!(
                    "Input at index {} size too large: {} (maximum: {})",
                    i, input_len, max_size
                )));
            }

            if input.iter().any(|&x| x.is_nan() || x.is_infinite()) {
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

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    status: &'static str,
    models_loaded: bool,
    redis_connected: bool,
}

pub async fn health_check(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let models_loaded = state.model_service.models_loaded();
    let redis_connected = state.model_service.redis_connected().await;
    let status = if models_loaded { "ok" } else { "error" };
    let code = if models_loaded {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    info!(
        models_loaded,
        redis_connected, status, "Health check requested"
    );
    (
        code,
        Json(HealthResponse {
            status,
            models_loaded,
            redis_connected,
        }),
    )
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
    metrics::record_inference_request();

    if let Err(err) = request.validate(
        config.min_input_size,
        config.max_input_size,
        config.max_batch_size,
    ) {
        metrics::record_inference_error();
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
            metrics::record_inference_error();
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
                let response = InferenceResponse {
                    output: vec![prediction],
                };
                (StatusCode::OK, Json(response)).into_response()
            }
            Err(e) => {
                metrics::record_inference_error();
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
        metrics::record_batch_request();
        match process_batch(
            model_service,
            request.input,
            request.model_version.clone(),
            request_id.clone(),
        )
        .await
        {
            Ok(predictions) => {
                let response = InferenceResponse {
                    output: predictions,
                };
                (StatusCode::OK, Json(response)).into_response()
            }
            Err(e) => {
                metrics::record_inference_error();
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

    metrics::record_inference_latency(timer.elapsed_seconds());

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
    let batch_timer = Timer::new();

    let outputs = model_service
        .infer_batch_with_version_with_request_id(
            inputs,
            model_version.as_deref(),
            request_id.as_deref(),
        )
        .await
        .map(|(values, _)| values);

    match &outputs {
        Ok(_) => {}
        Err(_) => metrics::record_inference_error(),
    }

    let duration = batch_timer.elapsed_seconds();
    info!(
        request_id = request_id.as_deref().unwrap_or("-"),
        batch_size,
        inference_duration_ms = (duration * 1000.0),
        "Batch inference completed"
    );
    metrics::record_inference_latency(duration);

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
