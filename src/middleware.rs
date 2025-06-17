use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::{
    body::Body,
    extract::{ConnectInfo, State},
    http::Request,
    middleware::Next,
    response::{IntoResponse, Response},
};
use tracing::{debug, warn};

use crate::error::AppError;
use crate::metrics;

pub struct RateLimiter {
    requests: Mutex<HashMap<IpAddr, Vec<Instant>>>,
    max_requests: u32,
    window_duration: Duration,
}

impl RateLimiter {
    pub fn new(max_requests: u32, window_seconds: u32) -> Self {
        Self {
            requests: Mutex::new(HashMap::new()),
            max_requests,
            window_duration: Duration::from_secs(window_seconds as u64),
        }
    }

    pub fn is_allowed(&self, ip: &IpAddr) -> bool {
        let now = Instant::now();
        let window_start = now - self.window_duration;

        let mut requests = self.requests.lock().unwrap();
        let timestamps = requests.entry(*ip).or_insert_with(Vec::new);

        timestamps.retain(|&timestamp| timestamp >= window_start);

        if timestamps.len() < self.max_requests as usize {
            timestamps.push(now);
            true
        } else {
            metrics::record_error("rate_limit_exceeded");
            metrics::record_rate_limit("exceeded");
            false
        }
    }

    pub fn cleanup(&self) {
        let now = Instant::now();
        let window_start = now - self.window_duration;

        let mut requests = self.requests.lock().unwrap();
        requests.retain(|_, timestamps| {
            timestamps.retain(|&timestamp| timestamp >= window_start);
            !timestamps.is_empty()
        });
    }
}

pub async fn rate_limit(
    ConnectInfo(addr): ConnectInfo<std::net::SocketAddr>,
    State(rate_limiter): State<Arc<RateLimiter>>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let ip = addr.ip();

    if rate_limiter.is_allowed(&ip) {
        debug!("Request allowed for IP: {}", ip);
        metrics::record_rate_limit("allowed");
        next.run(request).await
    } else {
        warn!("Rate limit exceeded for IP: {}", ip);
        metrics::record_api_request("rate_limited", "error");
        metrics::record_rate_limit("blocked");
        AppError::RateLimitExceeded.into_response()
    }
}

#[allow(dead_code)]
pub fn validation_error(message: &str) -> impl IntoResponse {
    metrics::record_error("validation_error");
    AppError::ValidationError(message.to_string()).into_response()
}
