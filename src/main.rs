use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tower_http::trace::TraceLayer;
use tract_core::prelude::*;
use tract_onnx::prelude::*;
use tracing::{info, Level};

use dotenvy::dotenv;

struct ModelService {
    model: Arc<Mutex<RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>>>,
}

impl ModelService {
    async fn new(model_path: &str) -> Result<Self> {
        info!("Loading model from {}", model_path);
        
        let model = tract_onnx::onnx()
            .model_for_path(model_path)
            .context("Failed to load model")?;
        
        let model = model
            .into_optimized()
            .context("Failed to optimize model")?;
        
        let model = model
            .into_runnable()
            .context("Failed to prepare model for running")?;
        
        Ok(Self {
            model: Arc::new(Mutex::new(model)),
        })
    }
    
    async fn infer(&self, input_data: Vec<f32>) -> Result<Vec<f32>> {
        let model = self.model.lock().await;
        
        let input = tract_ndarray::Array::from_shape_vec(
            (1, input_data.len()),
            input_data,
        )
        .context("Failed to create input tensor")?;
        
        let result = model
            .run(tvec!(input.into_tvalue()))
            .context("Failed to run inference")?;
        
        let output = result[0]
            .to_array_view::<f32>()
            .context("Failed to convert output to array")?;
        
        Ok(output.iter().copied().collect())
    }
}

#[derive(Debug, Deserialize)]
struct InferenceRequest {
    data: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct InferenceResponse {
    prediction: Vec<f32>,
}

async fn health_check() -> StatusCode {
    StatusCode::OK
}

async fn inference_handler(
    State(model_service): State<Arc<ModelService>>,
    Json(request): Json<InferenceRequest>,
) -> Result<Json<InferenceResponse>, StatusCode> {
    let prediction = model_service
        .infer(request.data)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    Ok(Json(InferenceResponse { prediction }))
}

#[tokio::main]
async fn main() -> Result<()> {

    dotenv().ok();

    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();
    
    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "model.onnx".to_string());
    
    if !Path::new(&model_path).exists() {
        anyhow::bail!("Model file not found: {}", model_path);
    }
    
    let model_service = Arc::new(
        ModelService::new(&model_path)
            .await
            .context("Failed to initialize model service")?,
    );
    
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/infer", post(inference_handler))
        .with_state(model_service)
        .layer(TraceLayer::new_for_http());
    
    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));
    info!("Server listening on {}", addr);
    
    let listener = tokio::net::TcpListener::bind(&addr).await.context("Failed to bind to address")?;
    info!("Server listening on {}", addr);
    
    axum::serve(listener, app)
        .await
        .context("Server error")?;
    
    Ok(())
}
