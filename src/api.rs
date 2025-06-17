use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tracing::{debug, error};

use crate::config::AppConfig;
use crate::error::AppError;
use crate::model::ModelService;
use crate::metrics::{self, Timer};

#[derive(Debug, Deserialize)]
pub struct InferenceRequest {
    input: Vec<f32>,
}

pub struct AppState {
    pub model_service: Arc<ModelService>,
    pub config: Arc<AppConfig>,
}


impl InferenceRequest {
    pub fn validate(&self, min_size: usize, max_size: usize) -> Result<(), AppError> {
        let input_len = self.input.len();
        
        if input_len < min_size {
            metrics::record_validation_error("input_too_small");
            return Err(AppError::ValidationError(
                format!("Input size too small: {} (minimum: {})", input_len, min_size)
            ));
        }
        
        if input_len > max_size {
            metrics::record_validation_error("input_too_large");
            return Err(AppError::ValidationError(
                format!("Input size too large: {} (maximum: {})", input_len, max_size)
            ));
        }
        
        if self.input.iter().any(|&x| x.is_nan() || x.is_infinite()) {
            metrics::record_validation_error("invalid_values");
            return Err(AppError::ValidationError(
                "Input contains NaN or infinity values".to_string()
            ));
        }
        
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct InferenceResponse {
    output: Vec<f32>,
}

pub async fn health_check() -> impl IntoResponse {
    debug!("Health check requested");
    metrics::record_api_request("health", "success");
    StatusCode::OK
}

pub async fn inference_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<InferenceRequest>,
) -> impl IntoResponse {
    let model_service = &state.model_service;
    let config = &state.config;
    debug!("Received inference request with input size: {}", request.input.len());
    
    let timer = Timer::new();
    metrics::record_api_request("inference", "started");
    
    if let Err(err) = request.validate(config.min_input_size, config.max_input_size) {
        metrics::record_api_request("inference", "validation_error");
        error!("Validation error: {}", err);
        return err.into_response();
    }

    let result = match model_service.infer(request.input).await {
        Ok(prediction) => {
            metrics::record_api_request("inference", "success");
            let response = InferenceResponse { output: prediction };
            (StatusCode::OK, Json(response)).into_response()
        }
        Err(e) => {
            metrics::record_api_request("inference", "error");
            metrics::record_error("api_inference");
            error!("Inference error: {}", e);
            
            let app_error = AppError::InferenceError(e.to_string());
            app_error.into_response()
        }
    };
    
    metrics::record_api_latency("inference", timer.elapsed_seconds());
    
    result
}

pub mod routes {
    use super::*;

    pub fn create_router(model_service: Arc<ModelService>, config: Arc<AppConfig>) -> Router {
        let state = Arc::new(AppState { model_service, config });
        
        Router::new()
            .route("/health", get(health_check))
            .route("/infer", post(inference_handler))
            .with_state(state)
    }
}
