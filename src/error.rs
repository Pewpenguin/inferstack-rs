use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use thiserror::Error;

#[allow(dead_code)]
#[derive(Debug, Error)]
pub enum AppError {
    #[error("{0}")]
    ValidationError(String),

    #[error("{0}")]
    ModelLoadError(String),
    #[error("{0}")]
    InferenceError(String),

    #[error("{0}")]
    CacheError(String),

    #[error("{0}")]
    ConfigError(String),

    #[error("{0}")]
    NotFound(String),

    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("{0}")]
    InternalError(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            Self::ValidationError(msg) => (StatusCode::UNPROCESSABLE_ENTITY, msg),
            Self::ModelLoadError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
            Self::InferenceError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
            Self::CacheError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
            Self::ConfigError(msg) => (StatusCode::BAD_REQUEST, msg),
            Self::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
            Self::RateLimitExceeded => (
                StatusCode::TOO_MANY_REQUESTS,
                "Rate limit exceeded. Please try again later.".to_string(),
            ),
            Self::InternalError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
        };

        (status, message).into_response()
    }
}
