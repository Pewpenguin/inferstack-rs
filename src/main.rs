use std::convert::TryInto;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use axum::Router;
use axum_prometheus::PrometheusMetricLayer;
use dotenvy::dotenv;
use tokio::signal;
use tower_http::trace::TraceLayer;
use tracing::{info, warn, Level};

use inferstack_rs::api::routes;
use inferstack_rs::cache::CacheService;
use inferstack_rs::config::AppConfig;
use inferstack_rs::middleware::RateLimiter;
use inferstack_rs::model::ModelService;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    let config = AppConfig::from_env().context("Invalid application configuration")?;

    config
        .validate_model_paths()
        .context("Startup aborted: model file validation failed")?;

    let cache_service = if let Some(redis_url) = &config.redis_url {
        info!("Initializing Redis cache with URL: {}", redis_url);

        let pool_size: u32 = config
            .redis_pool_size
            .try_into()
            .context("redis_pool_size does not fit into u32")?;

        if pool_size == 0 {
            anyhow::bail!("REDIS_POOL_SIZE must be greater than 0 when Redis caching is enabled");
        }

        info!(
            "Redis connection pool max_size (REDIS_POOL_SIZE): {}",
            pool_size
        );

        let service = CacheService::new_with_pool_size(redis_url, pool_size)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to initialize cache service: {}", e))?;

        match service.health_check().await {
            Ok(_) => info!("Redis connection health check successful"),
            Err(e) => warn!(
                "Redis health check failed: {}, but continuing with cache service",
                e
            ),
        }

        Some(Arc::new(service))
    } else {
        info!("No Redis URL provided, running without cache");
        None
    };

    let model_versions: Vec<(String, String, u8)> = config
        .model_versions
        .iter()
        .map(|v| (v.version.clone(), v.path.clone(), v.traffic_allocation))
        .collect();

    let model_service = Arc::new(
        ModelService::new_with_versions(
            model_versions,
            config.default_version.clone(),
            cache_service,
            config.cache_ttl,
            config.normalize_input,
            config.min_inference_ms_for_cache,
        )
        .await
        .map_err(|e| anyhow::anyhow!("Failed to initialize model service: {}", e))?,
    );

    let rate_limiter = Arc::new(RateLimiter::new(
        config.rate_limit_requests,
        config.rate_limit_window_secs,
    ));

    let (shutdown_tx, _) = tokio::sync::broadcast::channel::<()>(1);

    let rate_limiter_clone = rate_limiter.clone();
    let cleanup_shutdown_rx = shutdown_tx.subscribe();
    let cleanup_interval = config.rate_limit_cleanup_interval;
    info!("Rate limiter cleanup interval: {}s", cleanup_interval);

    let cleanup_task = tokio::spawn(async move {
        rate_limiter_clone
            .cleanup(cleanup_shutdown_rx, cleanup_interval)
            .await;
    });

    let shared_config = Arc::new(config);
    let (prometheus_layer, metrics_handler) = PrometheusMetricLayer::pair();

    let app = Router::new()
        .merge(routes::create_router(model_service, shared_config.clone()))
        .route(
            "/metrics",
            axum::routing::get(|| async move { metrics_handler.render() }),
        )
        .layer(prometheus_layer)
        .layer(TraceLayer::new_for_http())
        .layer(axum::middleware::from_fn_with_state(
            rate_limiter.clone(),
            inferstack_rs::middleware::rate_limit,
        ));

    let addr = SocketAddr::from(([0, 0, 0, 0], shared_config.port));
    info!("Server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .context("Failed to bind to address")?;

    let app = app.into_make_service_with_connect_info::<SocketAddr>();

    info!("Setting up graceful shutdown handler");
    let server = axum::serve(listener, app);

    let shutdown_signal = async move {
        let ctrl_c = async {
            signal::ctrl_c()
                .await
                .expect("Failed to install Ctrl+C handler");
        };

        #[cfg(unix)]
        let terminate = async {
            signal::unix::signal(signal::unix::SignalKind::terminate())
                .expect("Failed to install signal handler")
                .recv()
                .await;
        };

        #[cfg(not(unix))]
        let terminate = std::future::pending::<()>();

        tokio::select! {
            _ = ctrl_c => {},
            _ = terminate => {},
        }

        info!("Shutdown signal received, starting graceful shutdown");
        let _ = shutdown_tx.send(());
    };

    server
        .with_graceful_shutdown(shutdown_signal)
        .await
        .context("Server error")?;

    info!("Server has been gracefully shut down");

    if let Err(e) = cleanup_task.await {
        warn!("Error waiting for cleanup task to finish: {}", e);
    }

    tokio::time::sleep(Duration::from_millis(100)).await;

    Ok(())
}
