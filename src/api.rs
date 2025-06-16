use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tracing::{debug, error};

use crate::model::ModelService;
use crate::metrics::{self, Timer};

#[derive(Debug, Deserialize)]
pub struct InferenceRequest {
    input: Vec<f32>,
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
    State(model_service): State<Arc<ModelService>>,
    Json(request): Json<InferenceRequest>,
) -> impl IntoResponse {
    debug!("Received inference request");
    
    let timer = Timer::new();
    metrics::record_api_request("inference", "started");

    let result = match model_service.infer(request.input).await {
        Ok(prediction) => {
            metrics::record_api_request("inference", "success");
            let response = InferenceResponse { output: prediction };
            (StatusCode::OK, Json(response))
        }
        Err(e) => {
            metrics::record_api_request("inference", "error");
            metrics::record_error("api_inference");
            error!("Inference error: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, Json(InferenceResponse { output: vec![] }))
        }
    };
    
    metrics::record_api_latency("inference", timer.elapsed_seconds());
    
    result
}

pub mod routes {
    use super::*;

    pub fn create_router(model_service: Arc<ModelService>) -> Router {
        Router::new()
            .route("/health", get(health_check))
            .route("/infer", post(inference_handler))
            .with_state(model_service)
    }
}
