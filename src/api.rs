use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tracing::{error, info};

use crate::config::AppConfig;
use crate::error::AppError;
use crate::metrics::{self, Timer};
use crate::model::{InputSpec, ModelService, ModelSpec};

#[derive(Debug, Deserialize)]
pub struct InferenceRequest {
    #[serde(default)]
    pub input: Option<Vec<Vec<f32>>>,
    #[serde(default)]
    pub inputs: Option<HashMap<String, Vec<f32>>>,
    #[serde(default)]
    batch: bool,
    #[serde(default)]
    pub model_version: Option<String>,
}

pub struct AppState {
    pub model_service: Arc<ModelService>,
    pub config: Arc<AppConfig>,
}

impl InferenceRequest {
    fn schema_accepts_f32(dtype: &str) -> bool {
        let normalized = dtype.trim().to_ascii_uppercase();
        normalized == "F32" || normalized == "FLOAT32"
    }

    fn validate_tensor_dtype(spec: &InputSpec) -> Result<(), AppError> {
        if !Self::schema_accepts_f32(&spec.dtype) {
            return Err(AppError::ValidationError(format!(
                "Input '{}' expects dtype '{}' but request tensors are f32",
                spec.name, spec.dtype
            )));
        }
        Ok(())
    }

    fn expected_tensor_size(input_spec: &InputSpec) -> Option<usize> {
        if input_spec.shape.is_empty() {
            return None;
        }

        let dims = if input_spec.shape.len() > 1 {
            &input_spec.shape[1..]
        } else {
            &input_spec.shape[..]
        };

        if dims.is_empty() || dims.contains(&0) {
            return None;
        }

        Some(dims.iter().product())
    }

    fn validate_tensor_common(
        tensor: &[f32],
        input_name: &str,
        min_size: usize,
        max_size: usize,
    ) -> Result<(), AppError> {
        let input_len = tensor.len();
        if input_len < min_size {
            return Err(AppError::ValidationError(format!(
                "Input '{}' size too small: {} (minimum: {})",
                input_name, input_len, min_size
            )));
        }

        if input_len > max_size {
            return Err(AppError::ValidationError(format!(
                "Input '{}' size too large: {} (maximum: {})",
                input_name, input_len, max_size
            )));
        }

        if tensor.iter().any(|&x| x.is_nan() || x.is_infinite()) {
            return Err(AppError::ValidationError(format!(
                "Input '{}' contains NaN or infinity values",
                input_name
            )));
        }

        Ok(())
    }

    fn validate_against_schema(
        named_inputs: &HashMap<String, Vec<f32>>,
        model_spec: &ModelSpec,
    ) -> Result<(), AppError> {
        let required: HashSet<&str> = model_spec.inputs.iter().map(|s| s.name.as_str()).collect();
        for input_name in named_inputs.keys() {
            if !required.contains(input_name.as_str()) {
                return Err(AppError::ValidationError(format!(
                    "Unknown model input '{}'",
                    input_name
                )));
            }
        }

        for spec in &model_spec.inputs {
            if !named_inputs.contains_key(&spec.name) {
                return Err(AppError::ValidationError(format!(
                    "Missing required model input '{}'",
                    spec.name
                )));
            }
        }

        for spec in &model_spec.inputs {
            let value = named_inputs.get(&spec.name).ok_or_else(|| {
                AppError::ValidationError(format!("Missing required model input '{}'", spec.name))
            })?;
            Self::validate_tensor_dtype(spec)?;
            if let Some(expected) = Self::expected_tensor_size(spec) {
                if value.len() != expected {
                    return Err(AppError::ValidationError(format!(
                        "Input '{}' size mismatch: got {}, expected {}",
                        spec.name,
                        value.len(),
                        expected
                    )));
                }
            }
        }

        Ok(())
    }

    fn resolve_single_input(
        &self,
        model_spec: &ModelSpec,
        min_size: usize,
        max_size: usize,
    ) -> Result<Vec<f32>, AppError> {
        if let Some(named_inputs) = &self.inputs {
            if model_spec.inputs.len() != 1 {
                return Err(AppError::ValidationError(
                    "Named input inference currently supports models with exactly one input tensor"
                        .to_string(),
                ));
            }
            Self::validate_against_schema(named_inputs, model_spec)?;
            let first_name = &model_spec.inputs[0].name;
            let tensor = named_inputs
                .get(first_name)
                .ok_or_else(|| AppError::ValidationError(format!("Missing required model input '{}'", first_name)))?;
            Self::validate_tensor_common(tensor, first_name, min_size, max_size)?;
            return Ok(tensor.clone());
        }

        let legacy_inputs = self.input.as_ref().ok_or_else(|| {
            AppError::ValidationError(
                "Request must contain either 'inputs' (named tensors) or 'input'".to_string(),
            )
        })?;
        if legacy_inputs.len() != 1 {
            return Err(AppError::ValidationError(
                "Non-batch mode requires exactly one input".to_string(),
            ));
        }
        let first_input_spec = model_spec.inputs.first().ok_or_else(|| {
            AppError::ValidationError("Model has no input tensors".to_string())
        })?;
        Self::validate_tensor_dtype(first_input_spec)?;
        let tensor = &legacy_inputs[0];
        if let Some(expected) = Self::expected_tensor_size(first_input_spec) {
            if tensor.len() != expected {
                return Err(AppError::ValidationError(format!(
                    "Input '{}' size mismatch: got {}, expected {}",
                    first_input_spec.name,
                    tensor.len(),
                    expected
                )));
            }
        }
        Self::validate_tensor_common(tensor, &first_input_spec.name, min_size, max_size)?;
        Ok(tensor.clone())
    }

    fn resolve_batch_inputs(
        &self,
        model_spec: &ModelSpec,
        min_size: usize,
        max_size: usize,
        max_batch_size: usize,
    ) -> Result<Vec<Vec<f32>>, AppError> {
        if self.inputs.is_some() {
            return Err(AppError::ValidationError(
                "Named 'inputs' payload is currently only supported for non-batch inference"
                    .to_string(),
            ));
        }

        let legacy_inputs = self.input.as_ref().ok_or_else(|| {
            AppError::ValidationError(
                "Batch request must include legacy 'input' payload".to_string(),
            )
        })?;
        if legacy_inputs.is_empty() {
            return Err(AppError::ValidationError(
                "Batch cannot be empty".to_string(),
            ));
        }

        if legacy_inputs.len() > max_batch_size {
            return Err(AppError::ValidationError(format!(
                "Batch size too large: {} (maximum: 32)",
                legacy_inputs.len()
            )));
        }

        let first_input_spec = model_spec.inputs.first().ok_or_else(|| {
            AppError::ValidationError("Model has no input tensors".to_string())
        })?;
        Self::validate_tensor_dtype(first_input_spec)?;
        let expected = Self::expected_tensor_size(first_input_spec);

        for tensor in legacy_inputs {
            if let Some(exp) = expected {
                if tensor.len() != exp {
                    return Err(AppError::ValidationError(format!(
                        "Input '{}' size mismatch: got {}, expected {}",
                        first_input_spec.name,
                        tensor.len(),
                        exp
                    )));
                }
            }
            Self::validate_tensor_common(tensor, &first_input_spec.name, min_size, max_size)?;
        }

        Ok(legacy_inputs.clone())
    }

    pub fn validate(
        &self,
        model_spec: &ModelSpec,
        min_size: usize,
        max_size: usize,
        max_batch_size: usize,
    ) -> Result<Vec<Vec<f32>>, AppError> {
        if self.batch {
            self.resolve_batch_inputs(model_spec, min_size, max_size, max_batch_size)
        } else {
            self.resolve_single_input(model_spec, min_size, max_size)
                .map(|input| vec![input])
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum NamedOutputs {
    Single(HashMap<String, Vec<f32>>),
    Batch(Vec<HashMap<String, Vec<f32>>>),
}

#[derive(Debug, Serialize)]
pub struct InferenceResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    output: Option<Vec<Vec<f32>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    outputs: Option<NamedOutputs>,
}

#[derive(Debug, Serialize)]
pub struct TensorMetadata {
    name: String,
    shape: Vec<usize>,
    dtype: String,
}

#[derive(Debug, Serialize)]
pub struct ModelMetadata {
    version: String,
    inputs: Vec<TensorMetadata>,
    outputs: Vec<TensorMetadata>,
}

#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    models: Vec<ModelMetadata>,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    status: &'static str,
    models_loaded: bool,
    redis_connected: bool,
}

pub async fn health_check(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let models_loaded = state.model_service.models_loaded();
    let redis_connected = state.model_service.redis_connected().await;
    let status = if models_loaded { "ok" } else { "error" };
    let code = if models_loaded {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    info!(
        models_loaded,
        redis_connected, status, "Health check requested"
    );
    (
        code,
        Json(HealthResponse {
            status,
            models_loaded,
            redis_connected,
        }),
    )
}

fn normalize_dtype(dtype: &str) -> String {
    match dtype.trim().to_ascii_uppercase().as_str() {
        "F32" => "float32".to_string(),
        "F64" => "float64".to_string(),
        "I8" => "int8".to_string(),
        "I16" => "int16".to_string(),
        "I32" => "int32".to_string(),
        "I64" => "int64".to_string(),
        "U8" => "uint8".to_string(),
        "U16" => "uint16".to_string(),
        "U32" => "uint32".to_string(),
        "U64" => "uint64".to_string(),
        "BOOL" => "bool".to_string(),
        _ => dtype.to_ascii_lowercase(),
    }
}

fn build_tensor_metadata(spec: &InputSpec) -> TensorMetadata {
    TensorMetadata {
        name: spec.name.clone(),
        shape: spec.shape.clone(),
        dtype: normalize_dtype(&spec.dtype),
    }
}

pub async fn models_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let models = state
        .model_service
        .model_specs()
        .into_iter()
        .map(|(version, spec)| ModelMetadata {
            version,
            inputs: spec.inputs.iter().map(build_tensor_metadata).collect(),
            outputs: spec
                .outputs
                .iter()
                .map(|output| TensorMetadata {
                    name: output.name.clone(),
                    shape: output.shape.clone(),
                    dtype: normalize_dtype(&output.dtype),
                })
                .collect(),
        })
        .collect::<Vec<ModelMetadata>>();

    (StatusCode::OK, Json(ModelsResponse { models }))
}

pub async fn inference_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<InferenceRequest>,
) -> impl IntoResponse {
    let model_service = &state.model_service;
    let config = &state.config;
    let batch_size = request
        .input
        .as_ref()
        .map(|values| values.len())
        .or_else(|| request.inputs.as_ref().map(|_| 1))
        .unwrap_or(0);
    let request_id = headers
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let endpoint = if request.batch {
        "batch_inference"
    } else {
        "inference"
    };

    info!(
        endpoint,
        request_id = request_id.as_deref().unwrap_or("-"),
        batch_size,
        requested_model_version = request.model_version.as_deref().unwrap_or("auto"),
        "Inference request received"
    );

    let timer = Timer::new();
    metrics::record_inference_request();

    let model_spec = match model_service.model_spec_for_version(request.model_version.as_deref()) {
        Ok(spec) => spec,
        Err(err) => {
            metrics::record_inference_error();
            error!(
                endpoint,
                request_id = request_id.as_deref().unwrap_or("-"),
                error = %err,
                "Unable to resolve model schema"
            );
            return err.into_response();
        }
    };

    let resolved_inputs = match request.validate(
        model_spec,
        config.min_input_size,
        config.max_input_size,
        config.max_batch_size,
    ) {
        Ok(inputs) => inputs,
        Err(err) => {
            metrics::record_inference_error();
            error!(
                endpoint,
                request_id = request_id.as_deref().unwrap_or("-"),
                error = %err,
                "Request validation failed"
            );
            return err.into_response();
        }
    };
    let result = if !request.batch {
        let primary_output_name = model_spec.outputs.first().map(|output| output.name.clone());
        match model_service
            .infer_with_version_with_request_id(
                resolved_inputs[0].clone(),
                request.model_version.as_deref(),
                request_id.as_deref(),
            )
            .await
        {
            Ok((prediction, _executed_version)) => {
                let legacy_output = match primary_output_name.as_ref() {
                    Some(name) => match prediction.get(name) {
                        Some(values) => Some(vec![values.clone()]),
                        None => {
                            let err = AppError::InferenceError(format!(
                                "Primary output '{}' missing from inference result",
                                name
                            ));
                            error!(
                                endpoint,
                                request_id = request_id.as_deref().unwrap_or("-"),
                                error = %err,
                                "Inference response mapping failed"
                            );
                            return err.into_response();
                        }
                    },
                    None => None,
                };
                let response = InferenceResponse {
                    output: legacy_output,
                    outputs: Some(NamedOutputs::Single(prediction)),
                };
                (StatusCode::OK, Json(response)).into_response()
            }
            Err(e) => {
                metrics::record_inference_error();
                error!(
                    endpoint,
                    request_id = request_id.as_deref().unwrap_or("-"),
                    error = %e,
                    "Inference request failed"
                );
                e.into_response()
            }
        }
    } else {
        metrics::record_batch_request();
        let primary_output_name = model_spec.outputs.first().map(|output| output.name.clone());
        match process_batch(
            model_service,
            resolved_inputs,
            request.model_version.clone(),
            request_id.clone(),
        )
        .await
        {
            Ok(predictions) => {
                let legacy_output = match primary_output_name.as_ref() {
                    Some(name) => {
                        let mut legacy = Vec::with_capacity(predictions.len());
                        for prediction in &predictions {
                            match prediction.get(name) {
                                Some(values) => legacy.push(values.clone()),
                                None => {
                                    let err = AppError::InferenceError(format!(
                                        "Primary output '{}' missing from batch inference result",
                                        name
                                    ));
                                    error!(
                                        endpoint,
                                        request_id = request_id.as_deref().unwrap_or("-"),
                                        error = %err,
                                        "Batch inference response mapping failed"
                                    );
                                    return err.into_response();
                                }
                            }
                        }
                        Some(legacy)
                    }
                    None => None,
                };
                let response = InferenceResponse {
                    output: legacy_output,
                    outputs: Some(NamedOutputs::Batch(predictions)),
                };
                (StatusCode::OK, Json(response)).into_response()
            }
            Err(e) => {
                metrics::record_inference_error();
                error!(
                    endpoint,
                    request_id = request_id.as_deref().unwrap_or("-"),
                    error = %e,
                    "Batch inference request failed"
                );
                e.into_response()
            }
        }
    };

    metrics::record_inference_latency(timer.elapsed_seconds());

    result
}

async fn process_batch(
    model_service: &Arc<ModelService>,
    inputs: Vec<Vec<f32>>,
    model_version: Option<String>,
    request_id: Option<String>,
) -> Result<Vec<HashMap<String, Vec<f32>>>, AppError> {
    let batch_size = inputs.len();
    info!(
        request_id = request_id.as_deref().unwrap_or("-"),
        batch_size,
        requested_model_version = model_version.as_deref().unwrap_or("auto"),
        "Batch inference started"
    );
    let batch_timer = Timer::new();

    let outputs = model_service
        .infer_batch_with_version_with_request_id(
            inputs,
            model_version.as_deref(),
            request_id.as_deref(),
        )
        .await
        .map(|(values, _)| values);

    match &outputs {
        Ok(_) => {}
        Err(_) => metrics::record_inference_error(),
    }

    let duration = batch_timer.elapsed_seconds();
    info!(
        request_id = request_id.as_deref().unwrap_or("-"),
        batch_size,
        inference_duration_ms = (duration * 1000.0),
        "Batch inference completed"
    );
    metrics::record_inference_latency(duration);

    outputs
}

pub async fn batch_inference_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(mut request): Json<InferenceRequest>,
) -> impl IntoResponse {
    request.batch = true;
    inference_handler(State(state), headers, Json(request)).await
}

pub mod routes {
    use super::*;

    pub fn create_router(model_service: Arc<ModelService>, config: Arc<AppConfig>) -> Router {
        let state = Arc::new(AppState {
            model_service,
            config,
        });

        Router::new()
            .route("/health", get(health_check))
            .route("/models", get(models_handler))
            .route("/infer", post(inference_handler))
            .route("/batch", post(batch_inference_handler))
            .with_state(state)
    }
}
