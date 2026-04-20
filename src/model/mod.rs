use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, warn};
use tract_core::prelude::*;
use tract_onnx::prelude::*;

use crate::cache::CacheService;
use crate::config::NormalizeInput;
use crate::error::AppError;
use crate::metrics::{self, Timer};

mod inference;
mod preprocess;
mod routing;

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
    min_inference_ms_for_cache: u64,
}

impl ModelVersion {
    pub async fn new(
        version: String,
        model_path: &str,
        traffic_allocation: u8,
    ) -> Result<Self, AppError> {
        info!("Loading model version {} from {}", version, model_path);

        if !std::path::Path::new(model_path).exists() {
            return Err(AppError::ModelLoadError(format!(
                "Model file not found at path: {}",
                model_path
            )));
        }

        let model = tract_onnx::onnx().model_for_path(model_path).map_err(|e| {
            warn!("Failed to load model {}: {}", version, e);
            AppError::ModelLoadError(format!("Failed to load model version {}: {}", version, e))
        })?;

        info!("Model {} loaded successfully. Optimizing...", version);

        let model = model.into_optimized().map_err(|e| {
            warn!("Failed to optimize model {}: {}", version, e);
            AppError::ModelLoadError(format!(
                "Failed to optimize model version {}: {}",
                version, e
            ))
        })?;

        let model = model.into_runnable().map_err(|e| {
            warn!("Failed to prepare model {} for running: {}", version, e);
            AppError::ModelLoadError(format!(
                "Failed to prepare model version {} for running: {}",
                version, e
            ))
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
        min_inference_ms_for_cache: u64,
    ) -> Result<Self, AppError> {
        if model_configs.is_empty() {
            return Err(AppError::ConfigError(
                "At least one model configuration must be provided".to_string(),
            ));
        }

        let total_allocation: u16 = model_configs
            .iter()
            .map(|(_, _, allocation)| *allocation as u16)
            .sum();
        if total_allocation != 100 {
            return Err(AppError::ConfigError(format!(
                "Traffic allocations must sum to 100%, got {}%",
                total_allocation
            )));
        }

        let mut routing_table = Vec::with_capacity(model_configs.len());
        let mut prev_upper: u8 = 0;
        for (version, _, allocation) in &model_configs {
            let cumulative_upper = prev_upper
                .checked_add(*allocation)
                .ok_or_else(|| AppError::ConfigError("Traffic allocation overflow".to_string()))?;
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
                    return Err(AppError::ConfigError(format!(
                        "Default version {} not found in model configurations",
                        v
                    )));
                }
                v
            }
            None => routing_table
                .first()
                .map(|e| e.version.clone())
                .expect("routing_table non-empty when model_configs non-empty"),
        };

        info!(
            "Loaded {} model versions, default: {}",
            models.len(),
            default_version
        );

        Ok(Self {
            models,
            routing_table,
            default_version,
            cache,
            cache_ttl,
            normalize_input,
            min_inference_ms_for_cache,
        })
    }

    pub async fn infer_with_version(
        &self,
        input_data: Vec<f32>,
        version: Option<&str>,
    ) -> Result<(Vec<f32>, String), AppError> {
        metrics::record_batch_size(input_data.len());
        let timer = Timer::new();
        let mut cached = false;

        metrics::record_inference_request("started");

        let model_version: &ModelVersion = match self.select_model_version(version) {
            Ok(v) => v,
            Err(e) => {
                if version.is_some() {
                    metrics::record_inference_request("error");
                    metrics::record_error("model_selection");
                    return Err(e);
                }
                metrics::record_inference_request("error");
                metrics::record_error("model_selection");
                return Err(e);
            }
        };

        let (result, executed_version) = match self
            .infer_with_metrics(model_version, &input_data, &mut cached)
            .await
        {
            Ok(output) => (Ok(output), model_version.version.clone()),
            Err(e) => {
                warn!(
                    "Inference failed with model version {}: {}",
                    model_version.version, e
                );
                if model_version.version != self.default_version {
                    info!(
                        "Attempting fallback to default model version: {}",
                        self.default_version
                    );
                    if let Some(default_version) = self.models.get(&self.default_version) {
                        let fallback_result = self
                            .infer_with_metrics(default_version, &input_data, &mut cached)
                            .await;
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
}
