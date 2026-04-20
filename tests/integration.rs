mod common;

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use common::test_app::{build_test_app, get_status_body, post_json_status_body};
use serde::Deserialize;
use tower::ServiceExt;

#[derive(Debug, Deserialize)]
struct InferBody {
    output: Vec<Vec<f32>>,
}

#[tokio::test]
async fn test_health_endpoint() {
    let mut app = build_test_app().await.expect("app");
    let (status, _) = get_status_body(&mut app, "/health").await.expect("get");
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn test_infer_endpoint_success() {
    let mut app = build_test_app().await.expect("app");
    let body = serde_json::json!({ "input": [[1.0, 2.0, 3.0, 4.0]] });
    let (status, bytes) = post_json_status_body(&mut app, "/infer", &body)
        .await
        .expect("post");
    assert!(status.is_success());
    let parsed: InferBody = serde_json::from_slice(&bytes).expect("json");
    assert_eq!(parsed.output, vec![vec![1.0, 2.0, 3.0, 4.0]]);
}

#[tokio::test]
async fn test_batch_endpoint_success() {
    let mut app = build_test_app().await.expect("app");
    let body = serde_json::json!({
        "input": [[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 1.0]],
        "batch": true
    });
    let (status, bytes) = post_json_status_body(&mut app, "/batch", &body)
        .await
        .expect("post");
    assert!(status.is_success());
    let parsed: InferBody = serde_json::from_slice(&bytes).expect("json");
    assert_eq!(parsed.output.len(), 2);
}

#[tokio::test]
async fn test_invalid_payload() {
    let app = build_test_app().await.expect("app");
    let req = Request::builder()
        .method("POST")
        .uri("/infer")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from("not-json"))
        .expect("request");
    let res = app.clone().oneshot(req).await.expect("oneshot");
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
}
