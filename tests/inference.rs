mod common;

use common::test_app::{
    build_test_app, build_test_app_with, post_json_status_body, BuildTestAppOptions,
    BuildTestAppOverrides,
};
use inferstack_rs::config::ModelVersionConfig;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct InferBody {
    output: Vec<Vec<f32>>,
}

#[tokio::test]
async fn test_identity_model_inference() {
    let mut app = build_test_app().await.expect("app");
    let body = serde_json::json!({ "input": [[1.0, 2.0, 3.0, 4.0]] });
    let (status, bytes) = post_json_status_body(&mut app, "/infer", &body)
        .await
        .expect("req");
    assert!(status.is_success(), "{}", String::from_utf8_lossy(&bytes));
    let parsed: InferBody = serde_json::from_slice(&bytes).expect("json");
    assert_eq!(parsed.output[0], vec![1.0, 2.0, 3.0, 4.0]);
}

#[tokio::test]
async fn test_add_one_model_inference() {
    let mut app = build_test_app_with(
        BuildTestAppOptions {
            model_versions: vec![ModelVersionConfig {
                version: "v1".into(),
                path: common::test_app::fixture_path("add_one.onnx"),
                traffic_allocation: 100,
            }],
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

    let body = serde_json::json!({ "input": [[1.0, 2.0, 3.0, 4.0]] });
    let (status, bytes) = post_json_status_body(&mut app, "/infer", &body)
        .await
        .expect("req");
    assert!(status.is_success(), "{}", String::from_utf8_lossy(&bytes));
    let parsed: InferBody = serde_json::from_slice(&bytes).expect("json");
    assert_eq!(parsed.output[0], vec![2.0, 3.0, 4.0, 5.0]);
}

#[tokio::test]
async fn test_invalid_input_shape() {
    let mut app = build_test_app_with(
        BuildTestAppOptions {
            model_versions: vec![ModelVersionConfig {
                version: "v1".into(),
                path: common::test_app::fixture_path("identity.onnx"),
                traffic_allocation: 100,
            }],
            redis_url: None,
            cache_ttl: None,
            min_inference_ms_for_cache: 0,
            min_input_size: 4,
            max_input_size: 4,
        },
        BuildTestAppOverrides::default(),
    )
    .await
    .expect("app");

    let body = serde_json::json!({ "input": [[1.0, 2.0, 3.0]] });
    let (status, _) = post_json_status_body(&mut app, "/infer", &body)
        .await
        .expect("req");
    assert_eq!(status.as_u16(), 422);
}

#[tokio::test]
async fn test_batch_endpoint_basic() {
    let mut app = build_test_app().await.expect("app");
    let body = serde_json::json!({
        "input": [
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0]
        ],
        "batch": true
    });
    let (status, bytes) = post_json_status_body(&mut app, "/batch", &body)
        .await
        .expect("req");
    assert!(status.is_success(), "{}", String::from_utf8_lossy(&bytes));
    let parsed: InferBody = serde_json::from_slice(&bytes).expect("json");
    assert_eq!(parsed.output.len(), 2);
    assert_eq!(parsed.output[0], vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(parsed.output[1], vec![10.0, 20.0, 30.0, 40.0]);
}

#[tokio::test]
async fn test_batch_matches_multiple_single_inferences() {
    let mut app = build_test_app_with(
        BuildTestAppOptions {
            model_versions: vec![ModelVersionConfig {
                version: "v1".into(),
                path: common::test_app::fixture_path("add_one.onnx"),
                traffic_allocation: 100,
            }],
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

    let inputs = vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![10.0, 20.0, 30.0, 40.0],
        vec![-1.5, 0.0, 2.5, 8.0],
        vec![100.0, -50.0, 0.25, 7.75],
    ];

    let batch_body = serde_json::json!({
        "input": inputs,
        "batch": true,
        "model_version": "v1"
    });
    let (batch_status, batch_bytes) = post_json_status_body(&mut app, "/batch", &batch_body)
        .await
        .expect("batch req");
    assert!(
        batch_status.is_success(),
        "{}",
        String::from_utf8_lossy(&batch_bytes)
    );
    let batch_parsed: InferBody = serde_json::from_slice(&batch_bytes).expect("batch json");

    let mut single_outputs = Vec::with_capacity(batch_parsed.output.len());
    for input in batch_body["input"].as_array().expect("batch input array") {
        let infer_body = serde_json::json!({
            "input": [input],
            "model_version": "v1"
        });
        let (status, bytes) = post_json_status_body(&mut app, "/infer", &infer_body)
            .await
            .expect("single req");
        assert!(status.is_success(), "{}", String::from_utf8_lossy(&bytes));
        let parsed: InferBody = serde_json::from_slice(&bytes).expect("single json");
        assert_eq!(parsed.output.len(), 1);
        single_outputs.push(parsed.output[0].clone());
    }

    assert_eq!(batch_parsed.output.len(), single_outputs.len());
    for idx in 0..single_outputs.len() {
        assert_eq!(
            batch_parsed.output[idx], single_outputs[idx],
            "batch output mismatch at index {}",
            idx
        );
    }
}
