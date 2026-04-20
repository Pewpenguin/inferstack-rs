mod common;

use common::test_app::{
    build_test_app_with, fixture_path, post_json_status_body, BuildTestAppOptions, BuildTestAppOverrides,
};
use inferstack_rs::cache::CacheService;
use inferstack_rs::config::{ModelVersionConfig, NormalizeInput};
use serde::Deserialize;
use testcontainers::runners::AsyncRunner;
use testcontainers_modules::redis::Redis;

#[derive(Debug, Deserialize)]
struct InferBody {
    output: Vec<Vec<f32>>,
}

#[tokio::test]
#[ignore = "requires Docker for testcontainers Redis, or set REDIS_TEST_URL; run: cargo test --test cache -- --ignored"]
async fn test_cache_miss_then_hit() {
    let redis_url = match std::env::var("REDIS_TEST_URL") {
        Ok(url) => url,
        Err(_) => {
            let container = Redis::default()
                .start()
                .await
                .expect("start redis (Docker running?)");
            let host = container.get_host().await.expect("host");
            let port = container
                .get_host_port_ipv4(6379)
                .await
                .expect("port");
            format!("redis://{host}:{port}")
        }
    };

    let mut app = build_test_app_with(
        BuildTestAppOptions {
            model_versions: vec![ModelVersionConfig {
                version: "v1".into(),
                path: fixture_path("identity.onnx"),
                traffic_allocation: 100,
            }],
            redis_url: Some(redis_url),
            cache_ttl: Some(3600),
            min_inference_ms_for_cache: 0,
            min_input_size: 1,
            max_input_size: 1024,
        },
        BuildTestAppOverrides::default(),
    )
    .await
    .expect("app");

    let body = serde_json::json!({ "input": [[1.0, 2.0, 3.0, 4.0]] });
    let (status1, bytes1) = post_json_status_body(&mut app, "/infer", &body)
        .await
        .expect("first");
    assert!(status1.is_success(), "{}", String::from_utf8_lossy(&bytes1));
    let first: InferBody = serde_json::from_slice(&bytes1).expect("json");

    let (status2, bytes2) = post_json_status_body(&mut app, "/infer", &body)
        .await
        .expect("second");
    assert!(status2.is_success(), "{}", String::from_utf8_lossy(&bytes2));
    let second: InferBody = serde_json::from_slice(&bytes2).expect("json");

    assert_eq!(first.output, second.output);
}

#[test]
fn test_cache_key_consistency() {
    let k1 = CacheService::generate_key_with_version(
        "inference:v1",
        &vec![1.0f32, 2.0, 3.0, 4.0],
        1,
    )
    .expect("key");
    let k2 = CacheService::generate_key_with_version(
        "inference:v1",
        &vec![1.0f32, 2.0, 3.0, 4.0],
        1,
    )
    .expect("key");
    assert_eq!(k1, k2);
}

#[tokio::test]
async fn test_cache_disabled_behavior() {
    let mut app = build_test_app_with(
        BuildTestAppOptions {
            model_versions: vec![ModelVersionConfig {
                version: "v1".into(),
                path: fixture_path("identity.onnx"),
                traffic_allocation: 100,
            }],
            redis_url: None,
            cache_ttl: None,
            min_inference_ms_for_cache: 0,
            min_input_size: 1,
            max_input_size: 1024,
        },
        BuildTestAppOverrides {
            default_version: None,
            normalize_input: NormalizeInput::None,
        },
    )
    .await
    .expect("app");

    let body = serde_json::json!({ "input": [[1.0, 2.0, 3.0, 4.0]] });
    let (status, bytes) = post_json_status_body(&mut app, "/infer", &body)
        .await
        .expect("post");
    assert!(status.is_success(), "{}", String::from_utf8_lossy(&bytes));
}
