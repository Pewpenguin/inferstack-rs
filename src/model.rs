use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::sync::Mutex;
use tract_core::prelude::*;
use tract_onnx::prelude::*;
use tracing::{info, debug};

use crate::cache::CacheService;

pub struct ModelService {
    model: Arc<Mutex<RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>>>,
    cache: Option<Arc<CacheService>>,
    cache_ttl: Option<usize>,
}

impl ModelService {
    pub async fn new(model_path: &str, cache: Option<Arc<CacheService>>, cache_ttl: Option<usize>) -> Result<Self> {
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
            cache,
            cache_ttl,
        })
    }
    
    pub async fn infer(&self, input_data: Vec<f32>) -> Result<Vec<f32>> {
        // Check cache first if available
        if let Some(cache) = &self.cache {
            let cache_key = CacheService::generate_key("inference", &input_data)
                .context("Failed to generate cache key")?;
            
            if let Some(cached_result) = cache.get::<Vec<f32>>(&cache_key).await? {
                debug!("Using cached inference result");
                return Ok(cached_result);
            }
            
            // If not in cache, perform inference and then cache the result
            let result = self.perform_inference(input_data.clone()).await?;
            
            // Cache the result
            cache.set(&cache_key, &result, self.cache_ttl).await
                .context("Failed to cache inference result")?;
            
            Ok(result)
        } else {
            // No cache available, just perform inference
            self.perform_inference(input_data).await
        }
    }
    
    async fn perform_inference(&self, input_data: Vec<f32>) -> Result<Vec<f32>> {
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