mod common;

use std::sync::Arc;

use common::test_app::{
    self, build_test_app_with, fixture_path, post_json_status_body, BuildTestAppOptions,
    BuildTestAppOverrides,
};
use inferstack_rs::config::{ModelVersionConfig, NormalizeInput};
use inferstack_rs::model::ModelService;

async fn model_service_two_versions(alloc_a: u8, alloc_b: u8) -> ModelService {
    ModelService::new_with_versions(
        vec![
            ("v1".to_string(), fixture_path("identity.onnx"), alloc_a),
            ("v2".to_string(), fixture_path("add_one.onnx"), alloc_b),
        ],
        None,
        None,
        None,
        NormalizeInput::None,
        0,
    )
    .await
    .expect("load models")
}

#[tokio::test]
async fn test_single_version_routing() {
    let ms = model_service_two_versions(100, 0).await;
    let m = ms.select_model_version_with_roll(None, 0).expect("version");
    assert_eq!(m.version, "v1");
    let m2 = ms
        .select_model_version_with_roll(None, 99)
        .expect("version");
    assert_eq!(m2.version, "v1");
}

#[tokio::test]
async fn test_requested_version_override() {
    let ms = Arc::new(model_service_two_versions(50, 50).await);
    let (out, ver) = ms
        .infer_with_version(vec![1.0, 2.0, 3.0, 4.0], Some("v2"))
        .await
        .expect("infer v2");
    assert_eq!(ver, "v2");
    assert_eq!(out, vec![2.0, 3.0, 4.0, 5.0]);

    let (out2, ver2) = ms
        .infer_with_version(vec![1.0, 2.0, 3.0, 4.0], Some("v1"))
        .await
        .expect("infer v1");
    assert_eq!(ver2, "v1");
    assert_eq!(out2, vec![1.0, 2.0, 3.0, 4.0]);
}

#[tokio::test]
async fn test_invalid_version_error() {
    let mut app = test_app::build_test_app().await.expect("router");
    let body = serde_json::json!({
        "input": [[1.0, 2.0, 3.0, 4.0]],
        "model_version": "does-not-exist"
    });
    let (status, bytes) = post_json_status_body(&mut app, "/infer", &body)
        .await
        .expect("request");
    assert_eq!(status.as_u16(), 404);
    let text = String::from_utf8_lossy(&bytes);
    assert!(
        text.contains("not found") || text.contains("Not Found"),
        "unexpected body: {}",
        text
    );
}

#[tokio::test]
async fn test_weighted_distribution() {
    let ms = model_service_two_versions(50, 50).await;
    let mut c1 = 0u32;
    let mut c2 = 0u32;
    for r in 0u8..100 {
        let v = ms
            .select_model_version_with_roll(None, r)
            .expect("route")
            .version
            .clone();
        match v.as_str() {
            "v1" => c1 += 1,
            "v2" => c2 += 1,
            other => panic!("unexpected version {other}"),
        }
    }
    assert_eq!(c1, 50);
    assert_eq!(c2, 50);

    let ms2 = ModelService::new_with_versions(
        vec![
            ("a".into(), fixture_path("identity.onnx"), 30),
            ("b".into(), fixture_path("add_one.onnx"), 70),
        ],
        None,
        None,
        None,
        NormalizeInput::None,
        0,
    )
    .await
    .expect("load");
    let mut ca = 0;
    let mut cb = 0;
    for r in 0..100 {
        match ms2
            .select_model_version_with_roll(None, r)
            .expect("route")
            .version
            .as_str()
        {
            "a" => ca += 1,
            "b" => cb += 1,
            other => panic!("unexpected {other}"),
        }
    }
    assert_eq!(ca, 30);
    assert_eq!(cb, 70);
}

#[tokio::test]
async fn test_routing_http_reflects_version_override() {
    let mut app = build_test_app_with(
        BuildTestAppOptions {
            model_versions: vec![
                ModelVersionConfig {
                    version: "v1".into(),
                    path: fixture_path("identity.onnx"),
                    traffic_allocation: 50,
                },
                ModelVersionConfig {
                    version: "v2".into(),
                    path: fixture_path("add_one.onnx"),
                    traffic_allocation: 50,
                },
            ],
            redis_url: None,
            cache_ttl: None,
            min_inference_ms_for_cache: 0,
            min_input_size: 1,
            max_input_size: 1024,
        },
        BuildTestAppOverrides::default(),
    )
    .await
    .expect("app");

    let body = serde_json::json!({
        "input": [[1.0, 2.0, 3.0, 4.0]],
        "model_version": "v2"
    });
    let (status, bytes) = post_json_status_body(&mut app, "/infer", &body)
        .await
        .expect("req");
    assert!(status.is_success(), "{}", String::from_utf8_lossy(&bytes));
    let v: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
    assert_eq!(v["output"][0], serde_json::json!([2.0, 3.0, 4.0, 5.0]));
}
