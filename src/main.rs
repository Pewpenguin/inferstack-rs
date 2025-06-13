mod model;
mod api;
mod config;
mod cache;

use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use tracing::{info, Level};
use tower_http::trace::TraceLayer;
use dotenvy::dotenv;

use crate::model::ModelService;
use crate::api::routes;
use crate::config::AppConfig;
use crate::cache::CacheService;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();

    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();
    
    let config = AppConfig::from_env();
    
    if !Path::new(&config.model_path).exists() {
        anyhow::bail!("Model file not found: {}", config.model_path);
    }
    
    let cache_service = if let Some(redis_url) = &config.redis_url {
        info!("Initializing Redis cache with URL: {}", redis_url);
        let service = CacheService::new(redis_url)
            .context("Failed to initialize cache service")?;
        Some(Arc::new(service))
    } else {
        info!("No Redis URL provided, running without cache");
        None
    };
    
    let model_service = Arc::new(
        ModelService::new(
            &config.model_path,
            cache_service,
            config.cache_ttl
        )
        .await
        .context("Failed to initialize model service")?,
    );
    
    let app = routes::create_router(model_service)
        .layer(TraceLayer::new_for_http());
    
    let addr = SocketAddr::from(([0, 0, 0, 0], config.port));
    info!("Server listening on {}", addr);
    
    let listener = tokio::net::TcpListener::bind(&addr).await
        .context("Failed to bind to address")?;
    
    axum::serve(listener, app)
        .await
        .context("Server error")?;
    
    Ok(())
}
