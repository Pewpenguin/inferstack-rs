
pub struct ModelVersionConfig {
    pub version: String,
    pub path: String,
    pub traffic_allocation: u8,
}

pub struct AppConfig {
    pub model_path: String,
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
}

impl AppConfig {
    pub fn from_env() -> Self {
        dotenvy::dotenv().ok();

        let model_path = std::env::var("MODEL_PATH").unwrap_or_else(|_| "model.onnx".to_string());
        let port = std::env::var("PORT")
            .unwrap_or_else(|_| "3000".to_string())
            .parse()
            .unwrap_or(3000);
        let redis_url = std::env::var("REDIS_URL").ok();
        let cache_ttl = std::env::var("CACHE_TTL")
            .ok()
            .and_then(|v| v.parse().ok());
        let rate_limit_requests = std::env::var("RATE_LIMIT_REQUESTS")
            .unwrap_or_else(|_| "100".to_string())
            .parse()
            .unwrap_or(100);
        let rate_limit_window_secs = std::env::var("RATE_LIMIT_WINDOW_SECS")
            .unwrap_or_else(|_| "60".to_string())
            .parse()
            .unwrap_or(60);
        let max_input_size = std::env::var("MAX_INPUT_SIZE")
            .unwrap_or_else(|_| "1024".to_string())
            .parse()
            .unwrap_or(1024);
        let min_input_size = std::env::var("MIN_INPUT_SIZE")
            .unwrap_or_else(|_| "1".to_string())
            .parse()
            .unwrap_or(1);
        let redis_pool_size = std::env::var("REDIS_POOL_SIZE")
            .unwrap_or_else(|_| "5".to_string())
            .parse()
            .unwrap_or(5);
        let rate_limit_cleanup_interval = std::env::var("RATE_LIMIT_CLEANUP_INTERVAL")
            .unwrap_or_else(|_| "60".to_string())
            .parse()
            .unwrap_or(60);
        let max_batch_size = std::env::var("MAX_BATCH_SIZE")
            .unwrap_or_else(|_| "32".to_string())
            .parse()
            .unwrap_or(32);
        
        let default_version = std::env::var("DEFAULT_MODEL_VERSION").ok();
        
        let model_versions = std::env::var("MODEL_VERSIONS")
            .map(|versions_str| {
                versions_str
                    .split(',')
                    .filter_map(|version_str| {
                        let parts: Vec<&str> = version_str.splitn(4, ':').collect();
                        if parts.len() < 3 {
                            tracing::warn!("Invalid model version format: {}", version_str);
                            return None;
                        }
                        
                        let version = parts[0].trim().to_string();
                        let path = parts[1].trim().to_string();
                        let traffic_allocation = parts[2].trim().parse().unwrap_or_else(|_| {
                            tracing::warn!("Invalid traffic allocation for version {}: {}", version, parts[2]);
                            0
                        });
                        
                        Some(ModelVersionConfig {
                            version,
                            path,
                            traffic_allocation,
                        })
                    })
                    .collect()
            })
            .unwrap_or_else(|_| {
                vec![ModelVersionConfig {
                    version: "v1".to_string(),
                    path: model_path.clone(),
                    traffic_allocation: 100,
                }]
            });

        Self {
            model_path,
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
        }
    }
}
