use std::time::Instant;

use tracing::{debug, info, warn};
use tract_onnx::prelude::*;

use crate::cache::CacheService;
use crate::error::AppError;
use crate::metrics::{self, Timer};

use super::{ModelService, ModelVersion};

impl ModelService {
    pub(super) async fn infer_with_metrics(
        &self,
        model_version: &ModelVersion,
        input_data: &[f32],
        cached: &mut bool,
        request_id: Option<&str>,
    ) -> Result<Vec<f32>, AppError> {
        self.validate_input(model_version, input_data)?;

        let processed_input = self.preprocess_input(input_data)?;

        if let Some(cache) = &self.cache {
            let cache_key = CacheService::generate_key_with_version(
                &format!("inference:{}", model_version.version),
                &processed_input,
                1,
            )
            .map_err(|e| AppError::CacheError(format!("Failed to generate cache key: {}", e)))?;

            metrics::record_cache_operation("get", "attempt");

            match cache.get_with_retry::<Vec<f32>>(&cache_key, 2, 50).await {
                Ok(Some(cached_result)) => {
                    info!(
                        request_id = request_id.unwrap_or("-"),
                        model_version = %model_version.version,
                        cache_hit = true,
                        "Cache hit for inference result"
                    );
                    metrics::record_cache_operation("get", "hit");
                    *cached = true;
                    return Ok(cached_result);
                }
                Ok(None) => {
                    metrics::record_cache_operation("get", "miss");
                    info!(
                        request_id = request_id.unwrap_or("-"),
                        model_version = %model_version.version,
                        cache_hit = false,
                        "Cache miss for inference result"
                    );
                }
                Err(e) => {
                    metrics::record_cache_operation("get", "error");
                    metrics::record_error("cache_get");
                    warn!(
                        request_id = request_id.unwrap_or("-"),
                        model_version = %model_version.version,
                        error = %e,
                        "Cache lookup failed"
                    );
                }
            }

            let start = Instant::now();
            let result = self
                .perform_inference(model_version, processed_input.clone())
                .await?;
            let inference_time = start.elapsed().as_millis();
            info!(
                request_id = request_id.unwrap_or("-"),
                model_version = %model_version.version,
                inference_duration_ms = inference_time,
                "Model execution finished"
            );

            let should_write_cache = if self.min_inference_ms_for_cache == 0 {
                true
            } else {
                inference_time > u128::from(self.min_inference_ms_for_cache)
            };

            if should_write_cache {
                metrics::record_cache_operation("set", "attempt");

                let adaptive_ttl = self.cache_ttl.map(|base_ttl| {
                    if inference_time > 100 {
                        base_ttl * 2
                    } else if inference_time < 10 {
                        base_ttl / 2
                    } else {
                        base_ttl
                    }
                });

                match cache
                    .set_with_retry(&cache_key, &result, adaptive_ttl, 2, 50)
                    .await
                {
                    Ok(_) => {
                        metrics::record_cache_operation("set", "success");
                        debug!(
                            request_id = request_id.unwrap_or("-"),
                            model_version = %model_version.version,
                            ttl_seconds = ?adaptive_ttl,
                            "Cached inference result"
                        );
                    }
                    Err(e) => {
                        metrics::record_cache_operation("set", "error");
                        metrics::record_error("cache_set");
                        warn!(
                            request_id = request_id.unwrap_or("-"),
                            model_version = %model_version.version,
                            error = %e,
                            "Cache store failed"
                        );
                    }
                }

                if inference_time > 100 {
                    debug!(
                        "Would prefetch related inputs for expensive operation ({}ms)",
                        inference_time
                    );
                }
            } else {
                debug!(
                    "Skipping cache for fast inference ({}ms) for version {}",
                    inference_time, model_version.version
                );
            }

            Ok(result)
        } else {
            self.perform_inference(model_version, processed_input).await
        }
    }

    pub(super) async fn perform_inference(
        &self,
        model_version: &ModelVersion,
        input_data: Vec<f32>,
    ) -> Result<Vec<f32>, AppError> {
        let timer = Timer::new();

        let model = model_version.model.lock().await;

        let input = tract_ndarray::Array::from_shape_vec((1, input_data.len()), input_data)
            .map_err(|e| {
                AppError::InferenceError(format!("Failed to create input tensor: {}", e))
            })?;

        let result = model
            .run(tvec![input.into_tvalue()])
            .map_err(|e| AppError::InferenceError(format!("Failed to run inference: {}", e)))?;

        let output = result[0].to_array_view::<f32>().map_err(|e| {
            AppError::InferenceError(format!("Failed to convert output to array: {}", e))
        })?;

        metrics::record_model_execution_time_with_version(
            timer.elapsed_seconds(),
            &model_version.version,
        );

        Ok(output.iter().copied().collect())
    }
}
