use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::sync::Mutex;
use tract_core::prelude::*;
use tract_onnx::prelude::*;
use tracing::info;

pub struct ModelService {
    model: Arc<Mutex<RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>>>,
}

impl ModelService {
    pub async fn new(model_path: &str) -> Result<Self> {
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
    
    pub async fn infer(&self, input_data: Vec<f32>) -> Result<Vec<f32>> {
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