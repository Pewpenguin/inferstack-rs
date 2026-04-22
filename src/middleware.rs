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
use tokio::sync::broadcast;
use tracing::{debug, warn};

use crate::error::AppError;

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
        let timestamps = requests.entry(*ip).or_default();

        timestamps.retain(|&timestamp| timestamp >= window_start);

        if timestamps.len() < self.max_requests as usize {
            timestamps.push(now);
            true
        } else {
            false
        }
    }

    pub async fn cleanup(
        self: Arc<Self>,
        mut shutdown_rx: broadcast::Receiver<()>,
        cleanup_interval_secs: u64,
    ) {
        let cleanup_interval = Duration::from_secs(cleanup_interval_secs);
        debug!(
            "Starting rate limiter cleanup task with interval: {}s",
            cleanup_interval_secs
        );
        let mut interval = tokio::time::interval(cleanup_interval);

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    let start = Instant::now();
                    let window_start = Instant::now() - self.window_duration;
                    let mut requests = self.requests.lock().unwrap();
                    let initial_count = requests.len();

                    requests.retain(|_ip, timestamps| {
                        timestamps.retain(|&t| t >= window_start);
                        !timestamps.is_empty()
                    });

                    let removed = initial_count - requests.len();
                    let elapsed = start.elapsed();

                    if removed > 0 {
                        debug!(
                            "Rate limiter cleanup: removed {} expired entries, {} remaining (took {:?})",
                            removed, requests.len(), elapsed
                        );
                    } else if !requests.is_empty() {
                        debug!(
                            "Rate limiter cleanup: no expired entries, {} remaining (took {:?})",
                            requests.len(), elapsed
                        );
                    }
                }
                _ = shutdown_rx.recv() => {
                    debug!("Shutting down rate limiter cleanup task");
                    break;
                }
            }
        }
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
        next.run(request).await
    } else {
        warn!("Rate limit exceeded for IP: {}", ip);
        AppError::RateLimitExceeded.into_response()
    }
}

#[allow(dead_code)]
pub fn validation_error(message: &str) -> impl IntoResponse {
    AppError::ValidationError(message.to_string()).into_response()
}
