use std::path::Path;
use std::str::FromStr;

use tracing::{error, info};

use crate::error::AppError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormalizeInput {
    None,
    MinMax,
}

impl FromStr for NormalizeInput {
    type Err = AppError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "none" => Ok(Self::None),
            "minmax" => Ok(Self::MinMax),
            other => Err(AppError::ConfigError(format!(
                "Invalid NORMALIZE_INPUT value '{other}'. Allowed values: none, minmax"
            ))),
        }
    }
}

pub struct ModelVersionConfig {
    pub version: String,
    pub path: String,
    pub traffic_allocation: u8,
}

pub struct AppConfig {
    pub model_versions: Vec<ModelVersionConfig>,
    pub default_version: Option<String>,
    pub port: u16,
    pub redis_url: Option<String>,
    pub cache_ttl: Option<usize>,
    pub rate_limit_requests: u32,
    pub rate_limit_window_secs: u32,
    pub max_input_size: usize,
    pub min_input_size: usize,
    pub redis_pool_size: usize,
    pub rate_limit_cleanup_interval: u64,
    pub max_batch_size: usize,
    pub normalize_input: NormalizeInput,
    pub min_inference_ms_for_cache: u64,
}

impl AppConfig {
    pub fn from_env() -> Result<Self, AppError> {
        dotenvy::dotenv().ok();

        let default_model_path =
            std::env::var("MODEL_PATH").unwrap_or_else(|_| "model.onnx".to_string());

        fn parse_env_or_default<T>(
            key: &str,
            default: &str,
            validate: impl Fn(&T) -> Result<(), AppError>,
        ) -> Result<T, AppError>
        where
            T: std::str::FromStr,
            <T as std::str::FromStr>::Err: std::fmt::Display,
        {
            let raw = std::env::var(key).unwrap_or_else(|_| default.to_string());
            let parsed = raw.parse::<T>().map_err(|e| {
                AppError::ConfigError(format!(
                    "Invalid {key} value '{raw}': {e}. Expected a valid numeric value."
                ))
            })?;
            validate(&parsed)?;
            Ok(parsed)
        }

        let port: u16 = parse_env_or_default("PORT", "3000", |_| Ok(()))?;
        let redis_url = std::env::var("REDIS_URL").ok();

        let cache_ttl = match std::env::var("CACHE_TTL") {
            Ok(raw) => {
                let ttl = raw.parse::<usize>().map_err(|e| {
                    AppError::ConfigError(format!(
                        "Invalid CACHE_TTL value '{raw}': {e}. Expected a positive integer."
                    ))
                })?;
                if ttl == 0 {
                    return Err(AppError::ConfigError(
                        "Invalid CACHE_TTL value '0': expected a positive integer".to_string(),
                    ));
                }
                Some(ttl)
            }
            Err(_) => None,
        };

        let rate_limit_requests: u32 = parse_env_or_default("RATE_LIMIT_REQUESTS", "100", |v| {
            if *v == 0 {
                return Err(AppError::ConfigError(
                    "RATE_LIMIT_REQUESTS must be greater than 0".to_string(),
                ));
            }
            Ok(())
        })?;
        let rate_limit_window_secs: u32 =
            parse_env_or_default("RATE_LIMIT_WINDOW_SECS", "60", |v| {
                if *v == 0 {
                    return Err(AppError::ConfigError(
                        "RATE_LIMIT_WINDOW_SECS must be greater than 0".to_string(),
                    ));
                }
                Ok(())
            })?;
        let max_input_size: usize = parse_env_or_default("MAX_INPUT_SIZE", "1024", |v| {
            if *v == 0 {
                return Err(AppError::ConfigError(
                    "MAX_INPUT_SIZE must be greater than 0".to_string(),
                ));
            }
            Ok(())
        })?;
        let min_input_size: usize = parse_env_or_default("MIN_INPUT_SIZE", "1", |v| {
            if *v == 0 {
                return Err(AppError::ConfigError(
                    "MIN_INPUT_SIZE must be greater than 0".to_string(),
                ));
            }
            Ok(())
        })?;
        if min_input_size > max_input_size {
            return Err(AppError::ConfigError(format!(
                "Invalid input size limits: MIN_INPUT_SIZE ({min_input_size}) must be <= MAX_INPUT_SIZE ({max_input_size})"
            )));
        }

        let redis_pool_size: usize = parse_env_or_default("REDIS_POOL_SIZE", "5", |v| {
            if *v == 0 {
                return Err(AppError::ConfigError(
                    "REDIS_POOL_SIZE must be greater than 0".to_string(),
                ));
            }
            Ok(())
        })?;
        let rate_limit_cleanup_interval: u64 =
            parse_env_or_default("RATE_LIMIT_CLEANUP_INTERVAL", "60", |v| {
                if *v == 0 {
                    return Err(AppError::ConfigError(
                        "RATE_LIMIT_CLEANUP_INTERVAL must be greater than 0".to_string(),
                    ));
                }
                Ok(())
            })?;
        let max_batch_size: usize = parse_env_or_default("MAX_BATCH_SIZE", "32", |v| {
            if *v == 0 {
                return Err(AppError::ConfigError(
                    "MAX_BATCH_SIZE must be greater than 0".to_string(),
                ));
            }
            Ok(())
        })?;
        let min_inference_ms_for_cache: u64 =
            parse_env_or_default("MIN_INFERENCE_MS_FOR_CACHE", "5", |_| Ok(()))?;

        let normalize_input = match std::env::var("NORMALIZE_INPUT") {
            Ok(s) if s.trim().is_empty() => NormalizeInput::None,
            Ok(s) => s.parse()?,
            Err(_) => NormalizeInput::None,
        };

        let default_version = std::env::var("DEFAULT_MODEL_VERSION").ok();

        let model_versions = match std::env::var("MODEL_VERSIONS") {
            Ok(versions_str) => versions_str
                .split(',')
                .map(|version_str| {
                    let parts: Vec<&str> = version_str.splitn(4, ':').collect();
                    if parts.len() < 3 {
                        return Err(AppError::ConfigError(format!(
                            "Invalid MODEL_VERSIONS entry '{version_str}': expected format '<version>:<path>:<allocation>'"
                        )));
                    }

                    let version = parts[0].trim().to_string();
                    let path = parts[1].trim().to_string();
                    if version.is_empty() {
                        return Err(AppError::ConfigError(format!(
                            "Invalid MODEL_VERSIONS entry '{version_str}': version cannot be empty"
                        )));
                    }
                    if path.is_empty() {
                        return Err(AppError::ConfigError(format!(
                            "Invalid MODEL_VERSIONS entry '{version_str}': model path cannot be empty"
                        )));
                    }

                    let allocation_raw = parts[2].trim();
                    let traffic_allocation = allocation_raw.parse::<u8>().map_err(|e| {
                        AppError::ConfigError(format!(
                            "Invalid traffic allocation '{allocation_raw}' for model version '{version}': {e}"
                        ))
                    })?;

                    Ok(ModelVersionConfig {
                        version,
                        path,
                        traffic_allocation,
                    })
                })
                .collect::<Result<Vec<_>, AppError>>()?,
            Err(_) => vec![ModelVersionConfig {
                version: "v1".to_string(),
                path: default_model_path.clone(),
                traffic_allocation: 100,
            }],
        };

        if model_versions.is_empty() {
            return Err(AppError::ConfigError(
                "MODEL_VERSIONS did not contain any valid model entries".to_string(),
            ));
        }

        let total_allocation: u16 = model_versions
            .iter()
            .map(|mv| mv.traffic_allocation as u16)
            .sum();
        if total_allocation != 100 {
            return Err(AppError::ConfigError(format!(
                "MODEL_VERSIONS traffic allocations must sum to 100, got {total_allocation}"
            )));
        }

        Ok(Self {
            model_versions,
            default_version,
            port,
            redis_url,
            cache_ttl,
            rate_limit_requests,
            rate_limit_window_secs,
            max_input_size,
            min_input_size,
            redis_pool_size,
            rate_limit_cleanup_interval,
            max_batch_size,
            normalize_input,
            min_inference_ms_for_cache,
        })
    }

    pub fn test_with_models(
        model_versions: Vec<ModelVersionConfig>,
        default_version: Option<String>,
        redis_url: Option<String>,
        cache_ttl: Option<usize>,
        min_inference_ms_for_cache: u64,
        normalize_input: NormalizeInput,
        min_input_size: usize,
        max_input_size: usize,
    ) -> Self {
        Self {
            model_versions,
            default_version,
            port: 3000,
            redis_url,
            cache_ttl,
            rate_limit_requests: 10_000,
            rate_limit_window_secs: 60,
            max_input_size,
            min_input_size,
            redis_pool_size: 5,
            rate_limit_cleanup_interval: 60,
            max_batch_size: 32,
            normalize_input,
            min_inference_ms_for_cache,
        }
    }

    pub fn validate_model_paths(&self) -> Result<(), AppError> {
        if self.model_versions.is_empty() {
            error!("No model versions configured; set MODEL_VERSIONS or MODEL_PATH");
            return Err(AppError::ConfigError(
                "No model versions configured; set MODEL_VERSIONS or MODEL_PATH with at least one model"
                    .to_string(),
            ));
        }

        for mv in &self.model_versions {
            if !Path::new(&mv.path).exists() {
                error!(
                    "Model file not found for version {}: {}",
                    mv.version, mv.path
                );
                return Err(AppError::ConfigError(format!(
                    "Model file not found for version {}: {}",
                    mv.version, mv.path
                )));
            }
        }

        info!(
            "Validated {} configured model file(s) on disk",
            self.model_versions.len()
        );
        Ok(())
    }
}
