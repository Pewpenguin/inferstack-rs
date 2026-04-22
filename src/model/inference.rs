use std::time::Instant;

use tokio::task::spawn_blocking;
use tracing::{debug, info, warn};
use tract_onnx::prelude::*;

use crate::cache::CacheService;
use crate::error::AppError;
use crate::metrics;

use super::{ModelService, ModelVersion};

impl ModelService {
    fn expected_feature_count(model_version: &ModelVersion) -> Result<Option<usize>, AppError> {
        let input_spec = model_version
            .spec
            .inputs
            .first()
            .ok_or_else(|| AppError::ValidationError("Model has no input tensors".to_string()))?;

        if input_spec.shape.is_empty() {
            return Ok(None);
        }

        let dims = if input_spec.shape.len() > 1 {
            &input_spec.shape[1..]
        } else {
            &input_spec.shape[..]
        };

        if dims.is_empty() || dims.contains(&0) {
            return Ok(None);
        }

        Ok(Some(dims.iter().product()))
    }

    fn input_shape_for_batch(
        model_version: &ModelVersion,
        batch_size: usize,
        feature_count: usize,
    ) -> Result<Vec<usize>, AppError> {
        let expected = Self::expected_feature_count(model_version)?;
        let effective_features = expected.unwrap_or(feature_count);

        if feature_count != effective_features {
            return Err(AppError::ValidationError(format!(
                "Input tensor size mismatch: got {}, expected {}",
                feature_count, effective_features
            )));
        }

        let input_spec = model_version
            .spec
            .inputs
            .first()
            .ok_or_else(|| AppError::ValidationError("Model has no input tensors".to_string()))?;

        if input_spec.shape.len() <= 1 {
            return Ok(vec![batch_size, effective_features]);
        }

        let mut shape = Vec::with_capacity(input_spec.shape.len());
        shape.push(batch_size);
        for dim in &input_spec.shape[1..] {
            shape.push(if *dim == 0 { 1 } else { *dim });
        }
        Ok(shape)
    }

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

            match cache.get_with_retry::<Vec<f32>>(&cache_key, 2, 50).await {
                Ok(Some(cached_result)) => {
                    info!(
                        request_id = request_id.unwrap_or("-"),
                        model_version = %model_version.version,
                        cache_hit = true,
                        "Cache hit for inference result"
                    );
                    metrics::record_cache_hit();
                    *cached = true;
                    return Ok(cached_result);
                }
                Ok(None) => {
                    metrics::record_cache_miss();
                    info!(
                        request_id = request_id.unwrap_or("-"),
                        model_version = %model_version.version,
                        cache_hit = false,
                        "Cache miss for inference result"
                    );
                }
                Err(e) => {
                    metrics::record_inference_error();
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
                        debug!(
                            request_id = request_id.unwrap_or("-"),
                            model_version = %model_version.version,
                            ttl_seconds = ?adaptive_ttl,
                            "Cached inference result"
                        );
                    }
                    Err(e) => {
                        metrics::record_inference_error();
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
        let input_shape = Self::input_shape_for_batch(model_version, 1, input_data.len())?;
        let input = tract_ndarray::ArrayD::from_shape_vec(
            tract_ndarray::IxDyn(&input_shape),
            input_data,
        )
            .map_err(|e| {
                AppError::InferenceError(format!("Failed to create input tensor: {}", e))
            })?;

        let model = model_version.model.clone();
        let output_values = spawn_blocking(move || -> Result<Vec<f32>, AppError> {
            let result = model
                .run(tvec![input.into_tvalue()])
                .map_err(|e| AppError::InferenceError(format!("Failed to run inference: {}", e)))?;

            let output = result[0].to_array_view::<f32>().map_err(|e| {
                AppError::InferenceError(format!("Failed to convert output to array: {}", e))
            })?;

            Ok(output.iter().copied().collect())
        })
        .await
        .map_err(|e| AppError::InferenceError(format!("Inference task join error: {}", e)))??;
        Ok(output_values)
    }

    pub(super) async fn infer_batch_with_metrics(
        &self,
        model_version: &ModelVersion,
        inputs: Vec<Vec<f32>>,
    ) -> Result<Vec<Vec<f32>>, AppError> {
        let processed_inputs = inputs
            .iter()
            .map(|input| {
                self.validate_input(model_version, input)?;
                self.preprocess_input(input)
            })
            .collect::<Result<Vec<_>, _>>()?;

        if let Some(cache) = &self.cache {
            let cache_keys = processed_inputs
                .iter()
                .map(|input| {
                    CacheService::generate_key_with_version(
                        &format!("inference:{}", model_version.version),
                        input,
                        1,
                    )
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| {
                    AppError::CacheError(format!("Failed to generate batch cache keys: {}", e))
                })?;

            match cache.get_many::<Vec<f32>>(cache_keys.clone()).await {
                Ok(cached_results) => {
                    let mut outputs = vec![None; processed_inputs.len()];
                    let mut miss_indices = Vec::new();
                    let mut miss_inputs = Vec::new();
                    let mut miss_keys = Vec::new();

                    for idx in 0..cached_results.len() {
                        if let Some(value) = &cached_results[idx] {
                            metrics::record_cache_hit();
                            outputs[idx] = Some(value.clone());
                        } else {
                            metrics::record_cache_miss();
                            miss_indices.push(idx);
                            miss_inputs.push(processed_inputs[idx].clone());
                            miss_keys.push(cache_keys[idx].clone());
                        }
                    }

                    if !miss_inputs.is_empty() {
                        let start = Instant::now();
                        let miss_outputs = self
                            .run_batch_inference_with_fallback(model_version, miss_inputs)
                            .await?;
                        let inference_time = start.elapsed().as_millis();

                        let should_write_cache = if self.min_inference_ms_for_cache == 0 {
                            true
                        } else {
                            inference_time > u128::from(self.min_inference_ms_for_cache)
                        };

                        if should_write_cache {
                            let adaptive_ttl = self.cache_ttl.map(|base_ttl| {
                                if inference_time > 100 {
                                    base_ttl * 2
                                } else if inference_time < 10 {
                                    base_ttl / 2
                                } else {
                                    base_ttl
                                }
                            });

                            for idx in 0..miss_outputs.len() {
                                match cache
                                    .set_with_retry(
                                        &miss_keys[idx],
                                        &miss_outputs[idx],
                                        adaptive_ttl,
                                        2,
                                        50,
                                    )
                                    .await
                                {
                                    Ok(_) => {}
                                    Err(e) => {
                                        metrics::record_inference_error();
                                        warn!(error = %e, "Batch cache write failed");
                                    }
                                }
                            }
                        }

                        for idx in 0..miss_indices.len() {
                            outputs[miss_indices[idx]] = Some(miss_outputs[idx].clone());
                        }
                    }

                    let mut finalized = Vec::with_capacity(outputs.len());
                    for output in outputs {
                        finalized.push(output.ok_or_else(|| {
                            AppError::InferenceError(
                                "Missing output while assembling batch result".to_string(),
                            )
                        })?);
                    }
                    return Ok(finalized);
                }
                Err(e) => {
                    metrics::record_inference_error();
                    warn!(error = %e, "Batch cache lookup failed");
                }
            }
        }

        self.run_batch_inference_with_fallback(model_version, processed_inputs)
            .await
    }

    async fn run_batch_inference_with_fallback(
        &self,
        model_version: &ModelVersion,
        processed_inputs: Vec<Vec<f32>>,
    ) -> Result<Vec<Vec<f32>>, AppError> {
        match self
            .run_batch_inference(model_version, processed_inputs.clone())
            .await
        {
            Ok(outputs) => Ok(outputs),
            Err(AppError::InferenceError(msg))
                if msg.contains("Clashing resolution for expression") =>
            {
                let mut outputs = Vec::with_capacity(processed_inputs.len());
                for input in processed_inputs {
                    outputs.push(self.perform_inference(model_version, input).await?);
                }
                Ok(outputs)
            }
            Err(e) => Err(e),
        }
    }

    pub(super) async fn run_batch_inference(
        &self,
        model_version: &ModelVersion,
        inputs: Vec<Vec<f32>>,
    ) -> Result<Vec<Vec<f32>>, AppError> {
        if inputs.is_empty() {
            return Err(AppError::ValidationError(
                "Batch size must be greater than zero".to_string(),
            ));
        }

        let feature_count = inputs[0].len();
        if feature_count == 0 {
            return Err(AppError::ValidationError(
                "Input data cannot be empty".to_string(),
            ));
        }

        if inputs.iter().any(|input| input.len() != feature_count) {
            return Err(AppError::ValidationError(
                "All input vectors in a batch must have the same length".to_string(),
            ));
        }

        let batch_size = inputs.len();
        let flattened = inputs.into_iter().flatten().collect::<Vec<f32>>();
        let input_shape = Self::input_shape_for_batch(model_version, batch_size, feature_count)?;
        let input =
            tract_ndarray::ArrayD::from_shape_vec(tract_ndarray::IxDyn(&input_shape), flattened)
                .map_err(|e| {
                    AppError::InferenceError(format!(
                        "Failed to create batched input tensor: {}",
                        e
                    ))
                })?;

        let model = model_version.model.clone();
        let batched_output = spawn_blocking(move || -> Result<Vec<Vec<f32>>, AppError> {
            let result = model.run(tvec![input.into_tvalue()]).map_err(|e| {
                AppError::InferenceError(format!("Failed to run batch inference: {}", e))
            })?;

            let output = result[0].to_array_view::<f32>().map_err(|e| {
                AppError::InferenceError(format!("Failed to convert batch output to array: {}", e))
            })?;

            let output_shape = output.shape();
            if output_shape.is_empty() {
                return Err(AppError::InferenceError(
                    "Batch output tensor has no dimensions".to_string(),
                ));
            }

            if output_shape[0] != batch_size {
                return Err(AppError::InferenceError(format!(
                    "Batch output row count {} does not match input batch size {}",
                    output_shape[0], batch_size
                )));
            }

            let row_width = if output_shape.len() == 1 {
                1
            } else {
                output_shape[1..].iter().product::<usize>()
            };

            let values = output.iter().copied().collect::<Vec<f32>>();
            if values.len() != batch_size * row_width {
                return Err(AppError::InferenceError(format!(
                    "Batch output size {} does not match expected {}",
                    values.len(),
                    batch_size * row_width
                )));
            }

            Ok(values
                .chunks(row_width)
                .map(|chunk| chunk.to_vec())
                .collect::<Vec<Vec<f32>>>())
        })
        .await
        .map_err(|e| {
            AppError::InferenceError(format!("Batch inference task join error: {}", e))
        })??;
        Ok(batched_output)
    }
}
