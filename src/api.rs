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
    pub input: Vec<Vec<f32>>,
    #[serde(default)]
    batch: bool,
}

pub struct AppState {
    pub model_service: Arc<ModelService>,
    pub config: Arc<AppConfig>,
}


impl InferenceRequest {
    pub fn validate(&self, min_size: usize, max_size: usize, max_batch_size: usize) -> Result<(), AppError> {
        if self.input.is_empty() {
            metrics::record_validation_error("empty_batch");
            return Err(AppError::ValidationError(
                "Batch cannot be empty".to_string()
            ));
        }
        
        if self.input.len() > max_batch_size {
            metrics::record_validation_error("batch_too_large");
            return Err(AppError::ValidationError(
                format!("Batch size too large: {} (maximum: 32)", self.input.len())
            ));
        }
        
        for (i, input) in self.input.iter().enumerate() {
            let input_len = input.len();
            
            if input_len < min_size {
                metrics::record_validation_error("input_too_small");
                return Err(AppError::ValidationError(
                    format!("Input at index {} size too small: {} (minimum: {})", i, input_len, min_size)
                ));
            }
            
            if input_len > max_size {
                metrics::record_validation_error("input_too_large");
                return Err(AppError::ValidationError(
                    format!("Input at index {} size too large: {} (maximum: {})", i, input_len, max_size)
                ));
            }
            
            if input.iter().any(|&x| x.is_nan() || x.is_infinite()) {
                metrics::record_validation_error("invalid_values");
                return Err(AppError::ValidationError(
                    format!("Input at index {} contains NaN or infinity values", i)
                ));
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
    let batch_size = request.input.len();
    let endpoint = if request.batch { "batch_inference" } else { "inference" };
    
    debug!("Received {} request with batch size: {}", endpoint, batch_size);
    
    let timer = Timer::new();
    metrics::record_api_request(endpoint, "started");
    
    if let Err(err) = request.validate(config.min_input_size, config.max_input_size, config.max_batch_size) {
        metrics::record_api_request(endpoint, "validation_error");
        error!("Validation error: {}", err);
        return err.into_response();
    }
    
    let result = if !request.batch {
        if batch_size != 1 {
            metrics::record_api_request(endpoint, "validation_error");
            let err = AppError::ValidationError(
                "Non-batch mode requires exactly one input".to_string()
            );
            error!("Validation error: {}", err);
            return err.into_response();
        }
        
        match model_service.infer(request.input[0].clone()).await {
            Ok(prediction) => {
                metrics::record_api_request(endpoint, "success");
                let response = InferenceResponse { output: vec![prediction] };
                (StatusCode::OK, Json(response)).into_response()
            }
            Err(e) => {
                metrics::record_api_request(endpoint, "error");
                metrics::record_error("api_inference");
                error!("Inference error: {}", e);
                
                let app_error = AppError::InferenceError(e.to_string());
                app_error.into_response()
            }
        }
    } else {
        match process_batch(model_service, request.input).await {
            Ok(predictions) => {
                metrics::record_api_request(endpoint, "success");
                let response = InferenceResponse { output: predictions };
                (StatusCode::OK, Json(response)).into_response()
            }
            Err(e) => {
                metrics::record_api_request(endpoint, "error");
                metrics::record_error("api_batch_inference");
                error!("Batch inference error: {}", e);
                
                let app_error = AppError::InferenceError(e.to_string());
                app_error.into_response()
            }
        }
    };
    
    metrics::record_api_latency(endpoint, timer.elapsed_seconds());
    
    result
}

async fn process_batch(
    model_service: &Arc<ModelService>,
    inputs: Vec<Vec<f32>>
) -> anyhow::Result<Vec<Vec<f32>>> {
    use futures::future;
    use anyhow::Context;
    
    let batch_size = inputs.len();
    debug!("Processing batch of size: {}", batch_size);
    metrics::record_batch_size(batch_size);
    
    let batch_timer = Timer::new();
    
    let futures = inputs.into_iter().enumerate().map(|(idx, input)| {
        let model_service = model_service.clone();
        async move {
            metrics::record_batch_item("started");
            let result = model_service.infer(input).await
                .with_context(|| format!("Failed to process batch item at index {}", idx));
            
            match &result {
                Ok(_) => metrics::record_batch_item("success"),
                Err(_) => metrics::record_batch_item("error"),
            }
            
            result
        }
    });
    
    let results = future::join_all(futures).await;
    
    let mut outputs = Vec::with_capacity(results.len());
    for result in results {
        outputs.push(result?);
    }
    
    let duration = batch_timer.elapsed_seconds();
    debug!("Batch processing completed in {:.2}s", duration);
    metrics::record_api_latency("batch_processing", duration);
    metrics::record_batch_throughput(batch_size, duration);
    
    Ok(outputs)
}

pub async fn batch_inference_handler(
    State(state): State<Arc<AppState>>,
    Json(mut request): Json<InferenceRequest>,
) -> impl IntoResponse {
    request.batch = true;
    inference_handler(State(state), Json(request)).await
}

pub mod routes {
    use super::*;

    pub fn create_router(model_service: Arc<ModelService>, config: Arc<AppConfig>) -> Router {
        let state = Arc::new(AppState { model_service, config });
        
        Router::new()
            .route("/health", get(health_check))
            .route("/infer", post(inference_handler))
            .route("/batch", post(batch_inference_handler))
            .with_state(state)
    }
}
