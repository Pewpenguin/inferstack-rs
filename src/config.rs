pub struct AppConfig {
    pub model_path: String,
    pub port: u16,
    pub redis_url: Option<String>,
    pub cache_ttl: Option<usize>,
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

        Self {
            model_path,
            port,
            redis_url,
            cache_ttl,
        }
    }
}
