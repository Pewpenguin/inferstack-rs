use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result as AnyhowResult};
use rand::Rng;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};
use tract_core::prelude::*;
use tract_onnx::prelude::*;

use crate::cache::CacheService;
use crate::error::AppError;
use crate::metrics::{self, Timer};

type ModelType = RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

pub struct ModelVersion {
    pub version: String,
    model: Arc<Mutex<ModelType>>,
    pub traffic_allocation: u8,
}

pub struct ModelService {
    models: HashMap<String, ModelVersion>,
    default_version: String,
    cache: Option<Arc<CacheService>>,
    cache_ttl: Option<usize>,
}

impl ModelVersion {
    async fn new(
        version: String,
        model_path: &str,
        traffic_allocation: u8,
    ) -> AnyhowResult<Self> {
        info!("Loading model version {} from {}", version, model_path);

        let model = tract_onnx::onnx()
            .model_for_path(model_path)
            .context(format!("Failed to load model version {}", version))?;

        let model = model.into_optimized()
            .context(format!("Failed to optimize model version {}", version))?;

        let model = model
            .into_runnable()
            .context(format!("Failed to prepare model version {} for running", version))?;

        Ok(Self {
            version,
            model: Arc::new(Mutex::new(model)),
            traffic_allocation,
        })
    }
}

impl ModelService {  
    pub async fn new_with_versions(
        model_configs: Vec<(String, String, u8)>,
        default_version: Option<String>,
        cache: Option<Arc<CacheService>>,
        cache_ttl: Option<usize>,
    ) -> AnyhowResult<Self> {
        if model_configs.is_empty() {
            return Err(anyhow::anyhow!("At least one model configuration must be provided"));
        }
        
        let total_allocation: u16 = model_configs.iter().map(|(_, _, allocation)| *allocation as u16).sum();
        if total_allocation != 100 {
            return Err(anyhow::anyhow!("Traffic allocations must sum to 100%, got {}%", total_allocation));
        }
        
        let mut models = HashMap::new();
        
        for (version, path, allocation) in model_configs {
            let model_version = ModelVersion::new(
                version.clone(),
                &path,
                allocation,
            ).await?;
            
            models.insert(version.clone(), model_version);
        }
        
        let default_version = match default_version {
            Some(version) => {
                if !models.contains_key(&version) {
                    return Err(anyhow::anyhow!("Default version {} not found in model configurations", version));
                }
                version
            },
            None => models.keys().next().cloned().unwrap(), 
        };
        
        info!("Loaded {} model versions, default: {}", models.len(), default_version);
        
        Ok(Self {
            models,
            default_version,
            cache,
            cache_ttl,
        })
    }

    pub fn select_model_version(&self, requested_version: Option<&str>) -> Result<&ModelVersion, AppError> {
        if let Some(version) = requested_version {
            return self.models.get(version)
                .ok_or_else(|| AppError::ValidationError(format!("Model version '{}' not found", version)));
        }
        
        let mut rng = rand::rng();
        let random_value: u8 = rng.random_range(1..=100);
        
        let mut cumulative_allocation = 0;
        for model_version in self.models.values() {
            cumulative_allocation += model_version.traffic_allocation;
            if random_value <= cumulative_allocation {
                return Ok(model_version);
            }
        }
        
        self.models.get(&self.default_version)
            .ok_or_else(|| AppError::InternalError("Default model version not found".to_string()))
    }

    pub async fn infer_with_version(&self, input_data: Vec<f32>, version: Option<&str>) -> AnyhowResult<Vec<f32>> {
        metrics::record_batch_size(input_data.len());

        let timer = Timer::new();
        let mut cached = false;

        metrics::record_inference_request("started");
        
        let model_version = match self.select_model_version(version) {
            Ok(version) => version,
            Err(e) => {
                metrics::record_inference_request("error");
                metrics::record_error("model_selection");
                return Err(anyhow::anyhow!(e));
            }
        };
        
        metrics::record_model_version_usage(&model_version.version);

        let result = self.infer_with_metrics(model_version, input_data, &mut cached).await;

        metrics::record_inference_latency(timer.elapsed_seconds(), cached);

        match &result {
            Ok(_) => {
                metrics::record_inference_request("success");
                metrics::record_model_version_success(&model_version.version);
            },
            Err(_) => {
                metrics::record_inference_request("error");
                metrics::record_error("inference");
                metrics::record_model_version_error(&model_version.version);
            }
        }

        result
    }

    async fn infer_with_metrics(
        &self,
        model_version: &ModelVersion,
        input_data: Vec<f32>,
        cached: &mut bool,
    ) -> AnyhowResult<Vec<f32>> {
        if let Some(cache) = &self.cache {
            let cache_key = CacheService::generate_key_with_version(
                &format!("inference:{}", model_version.version),
                &input_data,
                1
            ).context("Failed to generate cache key")?;

            metrics::record_cache_operation("get", "attempt");

            let cache_result = cache.get_with_retry::<Vec<f32>>(&cache_key, 2, 50).await;

            match cache_result {
                Ok(Some(cached_result)) => {
                    debug!("Using cached inference result for version {}", model_version.version);
                    metrics::record_cache_operation("get", "hit");
                    *cached = true;
                    return Ok(cached_result);
                }
                Ok(None) => {
                    metrics::record_cache_operation("get", "miss");
                    debug!("Cache miss for key: {}", cache_key);
                }
                Err(e) => {
                    metrics::record_cache_operation("get", "error");
                    metrics::record_error("cache_get");
                    warn!("Cache get error: {}", e);
                }
            }

            let start = Instant::now();
            let result = self.perform_inference(model_version, input_data.clone()).await?;
            let inference_time = start.elapsed().as_millis();
            debug!("Model inference completed in {} ms for version {}", inference_time, model_version.version);

            if inference_time > 5 {
                metrics::record_cache_operation("set", "attempt");

                match cache
                    .set_with_retry(&cache_key, &result, self.cache_ttl, 2, 50)
                    .await
                {
                    Ok(_) => {
                        metrics::record_cache_operation("set", "success");
                        debug!("Successfully cached inference result for version {}", model_version.version);
                    }
                    Err(e) => {
                        metrics::record_cache_operation("set", "error");
                        metrics::record_error("cache_set");
                        warn!("Cache set error: {}", e);
                    }
                }
            } else {
                debug!("Skipping cache for fast inference ({}ms) for version {}", inference_time, model_version.version);
            }

            Ok(result)
        } else {
            self.perform_inference(model_version, input_data).await
        }
    }

    async fn perform_inference(&self, model_version: &ModelVersion, input_data: Vec<f32>) -> AnyhowResult<Vec<f32>> {
        let timer = Timer::new();

        let model = model_version.model.lock().await;

        let input = tract_ndarray::Array::from_shape_vec((1, input_data.len()), input_data)
            .context("Failed to create input tensor")?;

        let result = model
            .run(tvec![input.into_tvalue()])
            .context("Failed to run inference")?;

        let output = result[0]
            .to_array_view::<f32>()
            .context("Failed to convert output to array")?;

        metrics::record_model_execution_time_with_version(timer.elapsed_seconds(), &model_version.version);

        Ok(output.iter().copied().collect())
    }
}
