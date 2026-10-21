use super::*;
use axum::{
    body::{to_bytes, Body},
    http::Request,
};
use tower::ServiceExt;
fn mock() -> Service {
    Service::worker(Some("test-key".into()), |inputs, dims| {
        Ok((
            inputs
                .iter()
                .enumerate()
                .map(|(i, _)| vec![i as f32 + 0.5; dims])
                .collect(),
            inputs.len() * 3,
        ))
    })
}
async fn call(service: Service, body: &str, key: bool) -> (StatusCode, Value) {
    let mut request = Request::builder()
        .method("POST")
        .uri("/v1/embeddings")
        .header("content-type", "application/json");
    if key {
        request = request.header("authorization", "Bearer test-key");
    }
    let response = router(service)
        .oneshot(request.body(Body::from(body.to_owned())).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    (status, serde_json::from_slice(&body).unwrap())
}
#[tokio::test]
async fn ordered_batch_float_and_usage() {
    let (status, body) = call(
        mock(),
        r#"{"model":"google/embeddinggemma-2","input":["hello","world"],"dimensions":128}"#,
        true,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["data"][0]["index"], 0);
    assert_eq!(body["data"][1]["embedding"][0], 1.5);
    assert_eq!(body["data"][0]["embedding"].as_array().unwrap().len(), 128);
    assert_eq!(body["usage"]["prompt_tokens"], 6);
}
#[tokio::test]
async fn base64_is_little_endian_float32() {
    let (status, body) = call(mock(), r#"{"model":"embeddinggemma-2","input":"hello","dimensions":128,"encoding_format":"base64"}"#, true).await;
    assert_eq!(status, StatusCode::OK);
    let bytes = STANDARD
        .decode(body["data"][0]["embedding"].as_str().unwrap())
        .unwrap();
    assert_eq!(bytes.len(), 128 * 4);
    assert_eq!(&bytes[..4], &0.5f32.to_le_bytes());
}
#[tokio::test]
async fn errors_are_structured_and_do_not_echo_inputs() {
    for body in [
        r#"{"model":"wrong","input":"private-text"}"#,
        r#"{"model":"embeddinggemma-2","input":[]}"#,
        r#"{"model":"embeddinggemma-2","input":""}"#,
        r#"{"model":"embeddinggemma-2","input":"hello","dimensions":129}"#,
        r#"{"model":"embeddinggemma-2","input":"hello","encoding_format":"wrong"}"#,
        r#"{"model":"embeddinggemma-2","input":[262144]}"#,
        r#"{"model":"embeddinggemma-2","input":[-1]}"#,
        r#"{"model":"embeddinggemma-2","input":[1.5]}"#,
        "{",
    ] {
        let (status, value) = call(mock(), body, true).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert!(value["error"]["message"].is_string());
        assert!(!value.to_string().contains("private-text"));
    }
    assert_eq!(call(mock(), "{}", false).await.0, StatusCode::UNAUTHORIZED);
}
#[tokio::test]
async fn token_inputs_and_body_limit() {
    for input in [json!([1, 2, 3]), json!([[1, 2], [3, 4]])] {
        let body = json!({"model":MODEL_ID,"input":input}).to_string();
        assert_eq!(call(mock(), &body, true).await.0, StatusCode::OK);
    }
    let body = json!({"model":MODEL_ID,"input":"x".repeat(300000)}).to_string();
    assert_eq!(
        call(mock(), &body, true).await.0,
        StatusCode::PAYLOAD_TOO_LARGE
    );
}
#[tokio::test]
async fn queue_saturation_does_not_block_health() {
    let (tx, _rx) = mpsc::sync_channel(1);
    tx.try_send(Job {
        inputs: vec![],
        dims: 128,
        reply: oneshot::channel().0,
    })
    .unwrap();
    let service = Service {
        jobs: tx,
        api_key: Arc::new(None),
        alive: Arc::new(AtomicBool::new(true)),
    };
    assert_eq!(
        call(
            service.clone(),
            r#"{"model":"embeddinggemma-2","input":"hello"}"#,
            false
        )
        .await
        .0,
        StatusCode::SERVICE_UNAVAILABLE
    );
    let response = router(service)
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}
#[test]
fn batch_and_input_boundaries() {
    assert!(parse_input(json!(["x"])).is_ok());
    assert!(parse_input(json!(vec!["x"; 17])).is_err());
    assert!(parse_input(json!(" ")).is_err());
    assert!(parse_input(json!(vec![1; 1025])).is_err());
}

#[test]
fn internal_errors_are_distinct_from_input_errors() {
    assert_eq!(
        ApiError::encoding(crate::InvalidInput("Too many tokens").into()).0,
        StatusCode::BAD_REQUEST
    );
    assert_eq!(
        ApiError::encoding(anyhow::anyhow!("internal detail")).0,
        StatusCode::INTERNAL_SERVER_ERROR
    );
}
#[tokio::test]
async fn readiness_does_not_consume_queue_capacity() {
    let (tx, rx) = mpsc::sync_channel(1);
    let alive = Arc::new(AtomicBool::new(true));
    let service = Service {
        jobs: tx,
        api_key: Arc::new(None),
        alive: alive.clone(),
    };
    for _ in 0..20 {
        let response = router(service.clone())
            .oneshot(
                Request::builder()
                    .uri("/ready")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert!(matches!(rx.try_recv(), Err(mpsc::TryRecvError::Empty)));
    }
    alive.store(false, Ordering::Release);
    let response = router(service)
        .oneshot(
            Request::builder()
                .uri("/ready")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
}
