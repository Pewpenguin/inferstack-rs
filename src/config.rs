pub struct AppConfig {
    pub model_path: String,
    pub port: u16,
}

impl AppConfig {
    pub fn from_env() -> Self {
        let model_path = std::env::var("MODEL_PATH")
            .unwrap_or_else(|_| "model.onnx".to_string());
            
        let port = std::env::var("PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(3000);
            
        Self {
            model_path,
            port,
        }
    }
}