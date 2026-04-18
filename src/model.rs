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
use crate::config::NormalizeInput;
use crate::error::AppError;
use crate::metrics::{self, Timer};

type ModelType = RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;
pub struct RoutingEntry {
    pub version: String,
    #[allow(dead_code)] 
    pub allocation: u8,
    pub cumulative_upper: u8,
}

pub struct ModelVersion {
    pub version: String,
    model: Arc<Mutex<ModelType>>,
    #[allow(dead_code)]
    pub traffic_allocation: u8,
}

pub struct ModelService {
    models: HashMap<String, ModelVersion>,
    routing_table: Vec<RoutingEntry>,
    default_version: String,
    cache: Option<Arc<CacheService>>,
    cache_ttl: Option<usize>,
    normalize_input: NormalizeInput,
}

impl ModelVersion {
    pub async fn new(
        version: String,
        model_path: &str,
        traffic_allocation: u8,
    ) -> AnyhowResult<Self> {
        info!("Loading model version {} from {}", version, model_path);

        if !std::path::Path::new(model_path).exists() {
            return Err(anyhow::anyhow!("Model file not found at path: {}", model_path));
        }

        let model = tract_onnx::onnx()
            .model_for_path(model_path)
            .map_err(|e| {
                warn!("Failed to load model {}: {}", version, e);
                anyhow::anyhow!("Failed to load model version {}: {}", version, e)
            })?;

        info!("Model {} loaded successfully. Optimizing...", version);

        let model = model.into_optimized().map_err(|e| {
            warn!("Failed to optimize model {}: {}", version, e);
            anyhow::anyhow!("Failed to optimize model version {}: {}", version, e)
        })?;

        let model = model.into_runnable().map_err(|e| {
            warn!("Failed to prepare model {} for running: {}", version, e);
            anyhow::anyhow!("Failed to prepare model version {} for running: {}", version, e)
        })?;

        info!("Model version {} successfully prepared and ready", version);

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
        normalize_input: NormalizeInput,
    ) -> AnyhowResult<Self> {
        if model_configs.is_empty() {
            return Err(anyhow::anyhow!("At least one model configuration must be provided"));
        }

        let total_allocation: u16 = model_configs.iter().map(|(_, _, allocation)| *allocation as u16).sum();
        if total_allocation != 100 {
            return Err(anyhow::anyhow!("Traffic allocations must sum to 100%, got {}%", total_allocation));
        }

        let mut routing_table = Vec::with_capacity(model_configs.len());
        let mut prev_upper: u8 = 0;
        for (version, _, allocation) in &model_configs {
            let cumulative_upper = prev_upper
                .checked_add(*allocation)
                .ok_or_else(|| anyhow::anyhow!("Traffic allocation overflow"))?;
            routing_table.push(RoutingEntry {
                version: version.clone(),
                allocation: *allocation,
                cumulative_upper,
            });
            prev_upper = cumulative_upper;
        }
        debug_assert_eq!(prev_upper, 100);

        let mut models = HashMap::new();
        for (version, path, allocation) in model_configs {
            let model_version = ModelVersion::new(version.clone(), &path, allocation).await?;
            models.insert(version.clone(), model_version);
        }

        let default_version = match default_version {
            Some(v) => {
                if !models.contains_key(&v) {
                    return Err(anyhow::anyhow!("Default version {} not found in model configurations", v));
                }
                v
            }
            None => routing_table
                .first()
                .map(|e| e.version.clone())
                .expect("routing_table non-empty when model_configs non-empty"),
        };

        info!("Loaded {} model versions, default: {}", models.len(), default_version);

        Ok(Self {
            models,
            routing_table,
            default_version,
            cache,
            cache_ttl,
            normalize_input,
        })
    }

    pub fn select_model_version(&self, requested_version: Option<&str>) -> Result<&ModelVersion, AppError> {
        if let Some(version) = requested_version {
            return self.models.get(version)
                .ok_or_else(|| AppError::ValidationError(format!("Model version '{}' not found", version)));
        }

        let mut rng = rand::rng();
        let r = rng.random_range(0..100);

        let mut prev: u8 = 0;
        for entry in &self.routing_table {
            if (prev..entry.cumulative_upper).contains(&r) {
                return self
                    .models
                    .get(&entry.version)
                    .ok_or_else(|| AppError::InternalError("Routing entry missing loaded model".to_string()));
            }
            prev = entry.cumulative_upper;
        }

        self.models.get(&self.default_version)
            .ok_or_else(|| AppError::InternalError("Default model version not found".to_string()))
    }

    pub async fn infer_with_version(
        &self,
        input_data: Vec<f32>,
        version: Option<&str>,
    ) -> AnyhowResult<(Vec<f32>, String)> {
        metrics::record_batch_size(input_data.len());
        let timer = Timer::new();
        let mut cached = false;

        metrics::record_inference_request("started");

        let model_version: &ModelVersion = match self.select_model_version(version) {
            Ok(v) => v,
            Err(e) => {
                if version.is_some() {
                    let requested = version.unwrap_or("unknown");
                    warn!("Requested model version {} not found, falling back to default", requested);
                    metrics::FALLBACK_COUNTER.with_label_values(&[
                        requested,
                        &self.default_version,
                        "version_not_found"
                    ]).inc();

                    match self.models.get(&self.default_version) {
                        Some(default_version) => {
                            info!("Using fallback model version: {}", default_version.version);
                            default_version
                        }
                        None => {
                            metrics::record_inference_request("error");
                            metrics::record_error("model_selection");
                            return Err(anyhow::anyhow!(e));
                        }
                    }
                } else {
                    metrics::record_inference_request("error");
                    metrics::record_error("model_selection");
                    return Err(anyhow::anyhow!(e));
                }
            }
        };

        let (result, executed_version) = match self.infer_with_metrics(model_version, &input_data, &mut cached).await {
            Ok(output) => (Ok(output), model_version.version.clone()),
            Err(e) => {
                warn!("Inference failed with model version {}: {}", model_version.version, e);
                if model_version.version != self.default_version {
                    info!("Attempting fallback to default model version: {}", self.default_version);
                    if let Some(default_version) = self.models.get(&self.default_version) {
                        let fallback_result = self.infer_with_metrics(default_version, &input_data, &mut cached).await;
                        match fallback_result {
                            Ok(output) => {
                                info!("Fallback to default model version successful");
                                (Ok(output), default_version.version.clone())
                            }
                            Err(e2) => {
                                warn!("Fallback to default model version also failed");
                                (Err(e2), default_version.version.clone())
                            }
                        }
                    } else {
                        (Err(e), model_version.version.clone())
                    }
                } else {
                    (Err(e), model_version.version.clone())
                }
            }
        };

        metrics::record_inference_latency(timer.elapsed_seconds(), cached, &executed_version);

        match &result {
            Ok(_) => {
                metrics::record_inference_request("success");
                metrics::record_model_version_usage(&executed_version);
                metrics::record_model_version_success(&executed_version);
            }
            Err(_) => {
                metrics::record_inference_request("error");
                metrics::record_error("inference");
                metrics::record_model_version_error(&executed_version);
            }
        }

        result.map(|values| (values, executed_version))
    }

    fn validate_input(&self, _model_version: &ModelVersion, input_data: &[f32]) -> AnyhowResult<()> {
        if input_data.is_empty() {
            return Err(anyhow::anyhow!("Input data cannot be empty"));
        }
        Ok(())
    }

    fn preprocess_input(&self, input_data: &[f32]) -> AnyhowResult<Vec<f32>> {
        match self.normalize_input {
            NormalizeInput::None => Ok(input_data.to_vec()),
            NormalizeInput::MinMax => {
                // Opt-in only: legacy behavior scaled inputs when any component was outside [0,1].
                // That silently changed semantics; callers who need scaling must set NORMALIZE_INPUT=minmax.
                let mut processed = input_data.to_vec();

                let needs_normalization = processed.iter().copied().any(|x| x > 1.0 || x < 0.0);

                if needs_normalization {
                    debug!("Min–max normalizing input data to [0,1] range (NORMALIZE_INPUT=minmax)");

                    let min_val = processed.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max_val = processed.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let range = max_val - min_val;

                    if range > 0.0 {
                        for val in &mut processed {
                            *val = (*val - min_val) / range;
                        }
                    }
                }

                Ok(processed)
            }
        }
    }

    async fn infer_with_metrics(
        &self,
        model_version: &ModelVersion,
        input_data: &[f32],
        cached: &mut bool,
    ) -> AnyhowResult<Vec<f32>> {
        self.validate_input(model_version, input_data)?;

        let processed_input = self.preprocess_input(input_data)?;

        if let Some(cache) = &self.cache {
            let cache_key = CacheService::generate_key_with_version(
                &format!("inference:{}", model_version.version),
                &processed_input,
                1
            ).context("Failed to generate cache key")?;

            metrics::record_cache_operation("get", "attempt");

            match cache.get_with_retry::<Vec<f32>>(&cache_key, 2, 50).await {
                Ok(Some(cached_result)) => {
                    debug!("Using cached inference result for version {}", model_version.version);
                    metrics::record_cache_operation("get", "hit");
                    *cached = true;
                    return Ok(cached_result);
                },
                Ok(None) => {
                    metrics::record_cache_operation("get", "miss");
                    debug!("Cache miss for key: {}", cache_key);
                },
                Err(e) => {
                    metrics::record_cache_operation("get", "error");
                    metrics::record_error("cache_get");
                    warn!("Cache get error: {}", e);
                }
            }

            let start = Instant::now();
            let result = self.perform_inference(model_version, processed_input.clone()).await?;
            let inference_time = start.elapsed().as_millis();
            debug!("Model inference completed in {} ms for version {}", inference_time, model_version.version);

            if inference_time > 5 {
                metrics::record_cache_operation("set", "attempt");

                let adaptive_ttl = self.cache_ttl.map(|base_ttl| {
                    if inference_time > 100 { base_ttl * 2 }
                    else if inference_time < 10 { base_ttl / 2 }
                    else { base_ttl }
                });

                match cache.set_with_retry(&cache_key, &result, adaptive_ttl, 2, 50).await {
                    Ok(_) => {
                        metrics::record_cache_operation("set", "success");
                        debug!("Successfully cached inference result for version {} with TTL: {:?}s", model_version.version, adaptive_ttl);
                    }
                    Err(e) => {
                        metrics::record_cache_operation("set", "error");
                        metrics::record_error("cache_set");
                        warn!("Cache set error: {}", e);
                    }
                }

                if inference_time > 100 {
                    debug!("Would prefetch related inputs for expensive operation ({}ms)", inference_time);
                }
            } else {
                debug!("Skipping cache for fast inference ({}ms) for version {}", inference_time, model_version.version);
            }

            Ok(result)
        } else {
            self.perform_inference(model_version, processed_input).await
        }
    }

    async fn perform_inference(&self, model_version: &ModelVersion, input_data: Vec<f32>) -> AnyhowResult<Vec<f32>> {
        let timer = Timer::new();

        let model = model_version.model.lock().await;

        let input = tract_ndarray::Array::from_shape_vec((1, input_data.len()), input_data)
            .context("Failed to create input tensor")?;

        let result = model.run(tvec![input.into_tvalue()]).context("Failed to run inference")?;

        let output = result[0].to_array_view::<f32>().context("Failed to convert output to array")?;

        metrics::record_model_execution_time_with_version(timer.elapsed_seconds(), &model_version.version);

        Ok(output.iter().copied().collect())
    }
}
