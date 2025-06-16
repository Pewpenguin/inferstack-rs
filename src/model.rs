use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::sync::Mutex;
use tracing::{debug, info};
use tract_core::prelude::*;
use tract_onnx::prelude::*;

use crate::cache::CacheService;
use crate::metrics::{self, Timer};

pub struct ModelService {
    model:
        Arc<Mutex<RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>>>,
    cache: Option<Arc<CacheService>>,
    cache_ttl: Option<usize>,
}

impl ModelService {
    pub async fn new(
        model_path: &str,
        cache: Option<Arc<CacheService>>,
        cache_ttl: Option<usize>,
    ) -> Result<Self> {
        info!("Loading model from {}", model_path);

        let model = tract_onnx::onnx()
            .model_for_path(model_path)
            .context("Failed to load model")?;

        let model = model.into_optimized().context("Failed to optimize model")?;

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
        metrics::record_batch_size(input_data.len());
        
        let timer = Timer::new();
        let mut cached = false;
        
        metrics::record_inference_request("started");
        
        let result = self.infer_with_metrics(input_data, &mut cached).await;
        
        metrics::record_inference_latency(timer.elapsed_seconds(), cached);
        
        match &result {
            Ok(_) => metrics::record_inference_request("success"),
            Err(_) => {
                metrics::record_inference_request("error");
                metrics::record_error("inference");
            }
        }
        
        result
    }
    
    async fn infer_with_metrics(&self, input_data: Vec<f32>, cached: &mut bool) -> Result<Vec<f32>> {
        if let Some(cache) = &self.cache {
            let cache_key = CacheService::generate_key("inference", &input_data)
                .context("Failed to generate cache key")?;

            metrics::record_cache_operation("get", "attempt");
            
            match cache.get::<Vec<f32>>(&cache_key).await {
                Ok(Some(cached_result)) => {
                    debug!("Using cached inference result");
                    metrics::record_cache_operation("get", "hit");
                    *cached = true;
                    return Ok(cached_result);
                }
                Ok(None) => {
                    metrics::record_cache_operation("get", "miss");
                }
                Err(e) => {
                    metrics::record_cache_operation("get", "error");
                    metrics::record_error("cache_get");
                    debug!("Cache get error: {}", e);
                }
            }

            let result = self.perform_inference(input_data.clone()).await?;

            metrics::record_cache_operation("set", "attempt");
            match cache.set(&cache_key, &result, self.cache_ttl).await {
                Ok(_) => {
                    metrics::record_cache_operation("set", "success");
                }
                Err(e) => {
                    metrics::record_cache_operation("set", "error");
                    metrics::record_error("cache_set");
                    debug!("Cache set error: {}", e);
                }
            }

            Ok(result)
        } else {
            self.perform_inference(input_data).await
        }
    }

    async fn perform_inference(&self, input_data: Vec<f32>) -> Result<Vec<f32>> {
        let timer = Timer::new();
        
        let model = self.model.lock().await;

        let input = tract_ndarray::Array::from_shape_vec((1, input_data.len()), input_data)
            .context("Failed to create input tensor")?;

        let result = model
            .run(tvec!(input.into_tvalue()))
            .context("Failed to run inference")?;

        let output = result[0]
            .to_array_view::<f32>()
            .context("Failed to convert output to array")?;

        metrics::record_model_execution_time(timer.elapsed_seconds());
        
        Ok(output.iter().copied().collect())
    }
}
