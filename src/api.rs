use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};

use crate::model::ModelService;

#[derive(Debug, Deserialize)]
pub struct InferenceRequest {
    data: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct InferenceResponse {
    prediction: Vec<f32>,
}

pub async fn health_check() -> StatusCode {
    StatusCode::OK
}

pub async fn inference_handler(
    State(model_service): State<Arc<ModelService>>,
    Json(request): Json<InferenceRequest>,
) -> Result<Json<InferenceResponse>, StatusCode> {
    let prediction = model_service
        .infer(request.data)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    Ok(Json(InferenceResponse { prediction }))
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