use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use std::fmt;

#[allow(dead_code)]
#[derive(Debug)]
pub enum AppError {
    ValidationError(String),
    
    ModelLoadError(String),
    InferenceError(String),
    
    CacheError(String),
    
    RateLimitExceeded,
    
    InternalError(String),
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            Self::ModelLoadError(msg) => write!(f, "Model load error: {}", msg),
            Self::InferenceError(msg) => write!(f, "Inference error: {}", msg),
            Self::CacheError(msg) => write!(f, "Cache error: {}", msg),
            Self::RateLimitExceeded => write!(f, "Rate limit exceeded"),
            Self::InternalError(msg) => write!(f, "Internal server error: {}", msg),
        }
    }
}

impl std::error::Error for AppError {}

impl From<anyhow::Error> for AppError {
    fn from(err: anyhow::Error) -> Self {
        let err_string = err.to_string();
        
        if err_string.contains("validation") {
            Self::ValidationError(err_string)
        } else if err_string.contains("model") || err_string.contains("inference") {
            Self::InferenceError(err_string)
        } else if err_string.contains("cache") || err_string.contains("redis") {
            Self::CacheError(err_string)
        } else {
            Self::InternalError(err_string)
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            Self::ValidationError(msg) => (StatusCode::UNPROCESSABLE_ENTITY, msg),
            Self::ModelLoadError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
            Self::InferenceError(msg) => {
                if msg.contains("not found") {
                    (StatusCode::NOT_FOUND, msg)
                } else if msg.contains("invalid") {
                    (StatusCode::BAD_REQUEST, msg)
                } else {
                    (StatusCode::INTERNAL_SERVER_ERROR, msg)
                }
            },
            Self::CacheError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
            Self::RateLimitExceeded => (
                StatusCode::TOO_MANY_REQUESTS,
                "Rate limit exceeded. Please try again later.".to_string(),
            ),
            Self::InternalError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
        };
        
        (status, message).into_response()
    }
}