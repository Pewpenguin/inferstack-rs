use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use axum::Router;
use http_body_util::BodyExt;
use inferstack_rs::api::routes;
use inferstack_rs::cache::CacheService;
use inferstack_rs::config::{AppConfig, ModelVersionConfig, NormalizeInput};
use inferstack_rs::model::ModelService;
use serde::Serialize;
use tower::ServiceExt;

pub fn fixture_path(name: &str) -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
        .to_string_lossy()
        .into_owned()
}

pub async fn build_test_app() -> Result<Router> {
    build_test_app_with(
        BuildTestAppOptions::default(),
        BuildTestAppOverrides::default(),
    )
    .await
}

pub struct BuildTestAppOverrides {
    pub default_version: Option<String>,
    pub normalize_input: NormalizeInput,
}

impl Default for BuildTestAppOverrides {
    fn default() -> Self {
        Self {
            default_version: None,
            normalize_input: NormalizeInput::None,
        }
    }
}

pub struct BuildTestAppOptions {
    pub model_versions: Vec<ModelVersionConfig>,
    pub redis_url: Option<String>,
    pub cache_ttl: Option<usize>,
    pub min_inference_ms_for_cache: u64,
    pub min_input_size: usize,
    pub max_input_size: usize,
}

impl Default for BuildTestAppOptions {
    fn default() -> Self {
        Self {
            model_versions: vec![ModelVersionConfig {
                version: "v1".to_string(),
                path: fixture_path("identity.onnx"),
                traffic_allocation: 100,
            }],
            redis_url: None,
            cache_ttl: None,
            min_inference_ms_for_cache: 0,
            min_input_size: 1,
            max_input_size: 1024,
        }
    }
}

impl BuildTestAppOptions {
    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            !self.model_versions.is_empty(),
            "model_versions must not be empty"
        );
        let total: u16 = self
            .model_versions
            .iter()
            .map(|m| m.traffic_allocation as u16)
            .sum();
        anyhow::ensure!(total == 100, "traffic allocations must sum to 100%");
        for m in &self.model_versions {
            let p = PathBuf::from(&m.path);
            anyhow::ensure!(
                p.exists(),
                "model fixture missing at {} (run from crate root)",
                p.display()
            );
        }
        Ok(())
    }
}

pub async fn build_test_app_with(
    options: BuildTestAppOptions,
    overrides: BuildTestAppOverrides,
) -> Result<Router> {
    options.validate()?;

    let BuildTestAppOptions {
        model_versions,
        redis_url,
        cache_ttl,
        min_inference_ms_for_cache,
        min_input_size,
        max_input_size,
    } = options;

    let cache_service = if let Some(url) = &redis_url {
        Some(Arc::new(
            CacheService::new_with_pool_size(url, 4)
                .await
                .map_err(|e| anyhow::anyhow!(e))?,
        ))
    } else {
        None
    };

    let model_configs: Vec<(String, String, u8)> = model_versions
        .iter()
        .map(|m| (m.version.clone(), m.path.clone(), m.traffic_allocation))
        .collect();

    let model_service = Arc::new(
        ModelService::new_with_versions(
            model_configs,
            overrides.default_version.clone(),
            cache_service,
            cache_ttl,
            overrides.normalize_input,
            min_inference_ms_for_cache,
        )
        .await?,
    );

    let config = Arc::new(AppConfig::test_with_models(
        model_versions,
        overrides.default_version,
        redis_url,
        cache_ttl,
        min_inference_ms_for_cache,
        overrides.normalize_input,
        min_input_size,
        max_input_size,
    ));

    Ok(routes::create_router(model_service, config))
}

pub async fn post_json_response(
    app: &mut Router,
    uri: &str,
    body: &impl Serialize,
) -> Result<axum::http::Response<Body>> {
    let json = serde_json::to_vec(body)?;
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(json))?;
    Ok(app.oneshot(req).await?)
}

pub async fn response_body_bytes(res: axum::http::Response<Body>) -> Result<Vec<u8>> {
    let b = res.into_body().collect().await?.to_bytes();
    Ok(b.to_vec())
}

pub async fn post_json_status_body(
    app: &mut Router,
    uri: &str,
    body: &impl Serialize,
) -> Result<(StatusCode, Vec<u8>)> {
    let res = post_json_response(app, uri, body).await?;
    let status = res.status();
    let bytes = response_body_bytes(res).await?;
    Ok((status, bytes))
}

pub async fn get_status_body(app: &mut Router, uri: &str) -> Result<(StatusCode, Vec<u8>)> {
    let req = Request::builder().uri(uri).body(Body::empty())?;
    let res = app.oneshot(req).await?;
    let status = res.status();
    let bytes = response_body_bytes(res).await?;
    Ok((status, bytes))
}
