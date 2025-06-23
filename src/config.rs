pub struct AppConfig {
    pub model_path: String,
    pub port: u16,
    pub redis_url: Option<String>,
    pub cache_ttl: Option<usize>,
    pub rate_limit_requests: u32,
    pub rate_limit_window_secs: u32,
    pub max_input_size: usize,
    pub min_input_size: usize,
    pub redis_pool_size: usize,
    pub rate_limit_cleanup_interval: u64,
}

impl AppConfig {
    pub fn from_env() -> Self {
        let model_path = std::env::var("MODEL_PATH").unwrap_or_else(|_| "model.onnx".to_string());

        let port = std::env::var("PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(3000);

        let redis_url = std::env::var("REDIS_URL").ok();

        let cache_ttl = std::env::var("CACHE_TTL")
            .ok()
            .and_then(|ttl| ttl.parse().ok());

        let rate_limit_requests = std::env::var("RATE_LIMIT_REQUESTS")
            .ok()
            .and_then(|r| r.parse().ok())
            .unwrap_or(10);

        let rate_limit_window_secs = std::env::var("RATE_LIMIT_WINDOW_SECS")
            .ok()
            .and_then(|w| w.parse().ok())
            .unwrap_or(1);

        let max_input_size = std::env::var("MAX_INPUT_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10000);

        let min_input_size = std::env::var("MIN_INPUT_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);

        let redis_pool_size = std::env::var("REDIS_POOL_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(16);

        let rate_limit_cleanup_interval = std::env::var("RATE_LIMIT_CLEANUP_INTERVAL")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(60);

        Self {
            model_path,
            port,
            redis_url,
            cache_ttl,
            rate_limit_requests,
            rate_limit_window_secs,
            max_input_size,
            min_input_size,
            redis_pool_size,
            rate_limit_cleanup_interval,
        }
    }
}
