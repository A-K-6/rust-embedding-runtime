//! OpenAI-compatible embeddings transport with bounded serial inference.
use crate::{Input, Runtime, MODEL_ID};
use axum::{
    extract::{rejection::JsonRejection, DefaultBodyLimit, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use base64::{engine::general_purpose::STANDARD, Engine};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc,
};
use tokio::sync::oneshot;

type Embeddings = (Vec<Vec<f32>>, usize);
struct Job {
    inputs: Vec<Input>,
    dims: usize,
    reply: oneshot::Sender<Result<Embeddings, ApiError>>,
}
#[derive(Clone)]
pub struct Service {
    jobs: mpsc::SyncSender<Job>,
    api_key: Arc<Option<String>>,
    alive: Arc<AtomicBool>,
}
impl Service {
    pub fn new(mut runtime: Runtime, api_key: Option<String>) -> Self {
        Self::worker(api_key, move |inputs, dims| {
            runtime.embed(inputs, dims).map_err(ApiError::encoding)
        })
    }
    fn worker<F>(api_key: Option<String>, mut encode: F) -> Self
    where
        F: FnMut(&[Input], usize) -> Result<Embeddings, ApiError> + Send + 'static,
    {
        let (jobs, receive) = mpsc::sync_channel::<Job>(8);
        let alive = Arc::new(AtomicBool::new(true));
        let worker_alive = alive.clone();
        std::thread::spawn(move || {
            struct Liveness(Arc<AtomicBool>);
            impl Drop for Liveness {
                fn drop(&mut self) {
                    self.0.store(false, Ordering::Release);
                }
            }
            let _liveness = Liveness(worker_alive);
            while let Ok(job) = receive.recv() {
                if !job.reply.is_closed() {
                    let _ = job.reply.send(encode(&job.inputs, job.dims));
                }
            }
        });
        Self {
            jobs,
            api_key: Arc::new(api_key),
            alive,
        }
    }
    fn authorize(&self, headers: &HeaderMap) -> Result<(), ApiError> {
        if let Some(key) = self.api_key.as_ref() {
            let supplied = headers
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.strip_prefix("Bearer "));
            if supplied != Some(key.as_str()) {
                return Err(ApiError(
                    StatusCode::UNAUTHORIZED,
                    "Invalid API key",
                    "authentication_error",
                    None,
                ));
            }
        }
        Ok(())
    }
}

pub fn router(service: Service) -> Router {
    Router::new()
        .route("/health", get(|| async { Json(json!({"status":"ok"})) }))
        .route("/ready", get(ready))
        .route("/v1/models", get(models))
        .route("/v1/embeddings", post(embeddings))
        .fallback(|| async {
            ApiError(
                StatusCode::NOT_FOUND,
                "Route not found",
                "invalid_request_error",
                None,
            )
        })
        .method_not_allowed_fallback(|| async {
            ApiError(
                StatusCode::METHOD_NOT_ALLOWED,
                "Method not allowed",
                "invalid_request_error",
                None,
            )
        })
        .layer(DefaultBodyLimit::max(256 * 1024))
        .with_state(service)
}
async fn ready(State(state): State<Service>) -> Result<Json<Value>, ApiError> {
    if !state.alive.load(Ordering::Acquire) {
        return Err(ApiError(
            StatusCode::SERVICE_UNAVAILABLE,
            "Worker unavailable",
            "server_error",
            None,
        ));
    }
    Ok(Json(json!({"status":"ready"})))
}
async fn models(State(state): State<Service>, headers: HeaderMap) -> Result<Json<Value>, ApiError> {
    state.authorize(&headers)?;
    Ok(Json(
        json!({"object":"list","data":[{"id":MODEL_ID,"object":"model","created":0,"owned_by":"google"}]}),
    ))
}

#[derive(Deserialize)]
struct EmbedRequest {
    model: String,
    input: Value,
    #[serde(default = "default_dims")]
    dimensions: usize,
    encoding_format: Option<String>,
    // An optional OpenAI compatibility field; deliberately never persisted.
    #[serde(rename = "user")]
    _user: Option<String>,
}
fn default_dims() -> usize {
    768
}
fn parse_input(value: Value) -> Result<Vec<Input>, ApiError> {
    let bad = || {
        ApiError::bad(
            "Expected nonempty text, text array, token array, or token-array batch",
            "input",
        )
    };
    let tokens = |v: Vec<Value>| -> Result<Input, ApiError> {
        if v.is_empty() || v.len() > 1024 {
            return Err(bad());
        }
        let ids = v
            .into_iter()
            .map(|n| {
                n.as_u64()
                    .filter(|n| *n < 262144)
                    .map(|n| n as u32)
                    .ok_or_else(bad)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Input::Tokens(ids))
    };
    let inputs = match value {
        Value::String(s) => vec![Input::Text(s)],
        Value::Array(items) if !items.is_empty() => {
            if items.iter().all(Value::is_number) {
                vec![tokens(items)?]
            } else {
                if items.len() > 16 {
                    return Err(bad());
                }
                items
                    .into_iter()
                    .map(|v| match v {
                        Value::String(s) => Ok(Input::Text(s)),
                        Value::Array(ids) => tokens(ids),
                        _ => Err(bad()),
                    })
                    .collect::<Result<Vec<_>, _>>()?
            }
        }
        _ => return Err(bad()),
    };
    // Mixed text/token batches are an extension, but never accept empty strings.
    if inputs
        .iter()
        .any(|i| matches!(i, Input::Text(s) if s.trim().is_empty() || s.len() > 65536))
    {
        return Err(bad());
    }
    Ok(inputs)
}
async fn embeddings(
    State(state): State<Service>,
    headers: HeaderMap,
    payload: Result<Json<EmbedRequest>, JsonRejection>,
) -> Result<Json<Value>, ApiError> {
    state.authorize(&headers)?;
    let Json(request) = payload.map_err(|e| {
        ApiError(
            if e.status() == StatusCode::PAYLOAD_TOO_LARGE {
                StatusCode::PAYLOAD_TOO_LARGE
            } else {
                StatusCode::BAD_REQUEST
            },
            "Invalid JSON request or body limit exceeded",
            "invalid_request_error",
            None,
        )
    })?;
    if request.model != MODEL_ID && request.model != "embeddinggemma-2" {
        return Err(ApiError::bad(
            "Unsupported model; use google/embeddinggemma-2",
            "model",
        ));
    }
    if ![128, 256, 512, 768].contains(&request.dimensions) {
        return Err(ApiError::bad(
            "Dimensions must be 128, 256, 512, or 768",
            "dimensions",
        ));
    }
    let base64 = match request.encoding_format.as_deref().unwrap_or("float") {
        "float" => false,
        "base64" => true,
        _ => {
            return Err(ApiError::bad(
                "Encoding format must be float or base64",
                "encoding_format",
            ))
        }
    };
    let inputs = parse_input(request.input)?;
    let (reply, receive) = oneshot::channel();
    state
        .jobs
        .try_send(Job {
            inputs,
            dims: request.dimensions,
            reply,
        })
        .map_err(|_| {
            ApiError(
                StatusCode::SERVICE_UNAVAILABLE,
                "Inference queue full or worker unavailable",
                "server_error",
                None,
            )
        })?;
    let (vectors, tokens) = receive.await.map_err(|_| {
        ApiError(
            StatusCode::SERVICE_UNAVAILABLE,
            "Worker unavailable",
            "server_error",
            None,
        )
    })??;
    let data: Vec<_> = vectors
        .into_iter()
        .enumerate()
        .map(|(index, vector)| {
            let embedding = if base64 {
                Value::String(
                    STANDARD.encode(
                        vector
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect::<Vec<_>>(),
                    ),
                )
            } else {
                json!(vector)
            };
            json!({"object":"embedding", "index":index, "embedding":embedding})
        })
        .collect();
    Ok(Json(
        json!({"object":"list", "data":data, "model":MODEL_ID, "usage":{"prompt_tokens":tokens,"total_tokens":tokens}}),
    ))
}

pub struct ApiError(StatusCode, &'static str, &'static str, Option<&'static str>);
impl ApiError {
    fn encoding(error: anyhow::Error) -> Self {
        if let Some(error) = error.downcast_ref::<crate::InvalidInput>() {
            Self::bad(error.0, "input")
        } else {
            Self(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Encoder inference failed",
                "server_error",
                None,
            )
        }
    }
    fn bad(message: &'static str, param: &'static str) -> Self {
        Self(
            StatusCode::BAD_REQUEST,
            message,
            "invalid_request_error",
            Some(param),
        )
    }
}
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.0,
            Json(json!({"error":{"message":self.1,"type":self.2,"param":self.3,"code":null}})),
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests;
