mod api;
mod cache;
mod config;
mod metrics;
mod model;

use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use axum::Router;
use axum_prometheus::PrometheusMetricLayer;
use dotenvy::dotenv;
use tower_http::trace::TraceLayer;
use tracing::{info, Level};

use crate::api::routes;
use crate::cache::CacheService;
use crate::config::AppConfig;
use crate::model::ModelService;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();

    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    let config = AppConfig::from_env();

    if !Path::new(&config.model_path).exists() {
        anyhow::bail!("Model file not found: {}", config.model_path);
    }

    let cache_service = if let Some(redis_url) = &config.redis_url {
        info!("Initializing Redis cache with URL: {}", redis_url);
        let service = CacheService::new(redis_url)
            .await
            .context("Failed to initialize cache service")?;
        Some(Arc::new(service))
    } else {
        info!("No Redis URL provided, running without cache");
        None
    };

    let model_service = Arc::new(
        ModelService::new(&config.model_path, cache_service, config.cache_ttl)
            .await
            .context("Failed to initialize model service")?,
    );

    let (prometheus_layer, metrics_handler) = PrometheusMetricLayer::pair();
    
    let app = Router::new()
        .merge(routes::create_router(model_service))
        .route("/metrics", axum::routing::get(|| async move { metrics_handler.render() }))
        .layer(prometheus_layer)
        .layer(TraceLayer::new_for_http());

    let addr = SocketAddr::from(([0, 0, 0, 0], config.port));
    info!("Server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .context("Failed to bind to address")?;

    axum::serve(listener, app).await.context("Server error")?;

    Ok(())
}
