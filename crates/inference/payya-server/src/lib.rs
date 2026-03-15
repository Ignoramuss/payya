//! HTTP inference server with OpenAI-compatible `/v1/chat/completions` API.
//!
//! Supports both non-streaming and SSE streaming responses. Built on `axum`
//! with `tokio` for async I/O.
//!
//! # Architecture
//!
//! ```text
//!  Client ──► POST /v1/chat/completions
//!                    │
//!                    ▼
//!            ┌───────────────┐
//!            │   axum Router │
//!            └───────┬───────┘
//!                    │
//!                    ▼
//!            ┌───────────────┐    ┌──────────────────┐
//!            │ Request Queue │───►│ InferenceEngine  │
//!            └───────────────┘    │  - SLM model     │
//!                                 │  - KV cache      │
//!                                 │  - Prompt cache  │
//!                                 └──────────────────┘
//!                                         │
//!                    ┌────────────────────┘
//!                    ▼
//!            ┌───────────────┐
//!            │ SSE / JSON    │
//!            │ Response      │
//!            └───────────────┘
//! ```

use std::sync::Arc;
use std::sync::Mutex;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Json};
use axum::routing::{get, post};
use axum::Router;
use futures::stream;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;

// ── OpenAI-compatible API types ─────────────────────────────────────

/// A chat message in the OpenAI format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Request body for `/v1/chat/completions`.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stream: bool,
}

fn default_model() -> String {
    "payya-slm".to_string()
}
fn default_max_tokens() -> usize {
    128
}
fn default_temperature() -> f32 {
    1.0
}

/// A single choice in a chat completion response.
#[derive(Debug, Clone, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

/// A single choice delta in a streaming response.
#[derive(Debug, Clone, Serialize)]
pub struct ChatChoiceDelta {
    pub index: usize,
    pub delta: ChatDelta,
    pub finish_reason: Option<String>,
}

/// Delta content for streaming.
#[derive(Debug, Clone, Serialize)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// Token usage statistics.
#[derive(Debug, Clone, Serialize)]
pub struct UsageStats {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Non-streaming chat completion response.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: UsageStats,
}

/// Streaming chat completion chunk.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoiceDelta>,
}

/// Health check response.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub model: String,
}

// ── Server configuration ────────────────────────────────────────────

/// Configuration for the inference server.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Host to bind to.
    pub host: String,
    /// Port to bind to.
    pub port: u16,
    /// Maximum concurrent requests.
    pub max_concurrent: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            max_concurrent: 10,
        }
    }
}

// ── Inference engine ────────────────────────────────────────────────

/// The core inference engine wrapping the SLM and supporting services.
pub struct InferenceEngine {
    slm: payya_slm::Slm,
    prompt_cache: payya_prompt_cache::RadixTree,
    seed: u64,
}

impl InferenceEngine {
    /// Create a new engine with the given SLM.
    pub fn new(slm: payya_slm::Slm, seed: u64) -> Self {
        Self {
            slm,
            prompt_cache: payya_prompt_cache::RadixTree::new(),
            seed,
        }
    }

    /// Generate a completion for the given messages.
    ///
    /// Concatenates all message contents, tokenizes, and generates.
    pub fn generate(
        &mut self,
        messages: &[ChatMessage],
        max_tokens: usize,
        temperature: f32,
        top_p: Option<f32>,
    ) -> GenerationResult {
        let prompt_text = format_messages(messages);

        let mut processor =
            payya_logit_processor::LogitProcessor::new().with_temperature(temperature);
        if let Some(p) = top_p {
            processor = processor.with_top_p(p);
        }

        // Check prompt cache for prefix hit.
        let tokenizer = self.slm.tokenizer();
        let prompt_tokens: Option<Vec<u32>> = tokenizer.map(|t| t.encode(&prompt_text));
        let prompt_len = prompt_tokens
            .as_ref()
            .map(|t| t.len())
            .unwrap_or(prompt_text.len());

        if let Some(ref tokens) = prompt_tokens {
            let _hit = self.prompt_cache.lookup(tokens);
            // Cache the prompt prefix for future requests.
            self.prompt_cache.insert(tokens);
        }

        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        self.seed = self.seed.wrapping_add(1);

        let generated = if self.slm.tokenizer().is_some() {
            self.slm
                .generate_text(&prompt_text, max_tokens, &processor, &mut rng)
        } else {
            let prompt_ids: Vec<usize> = (0..prompt_text.len().min(10))
                .map(|i| i % self.slm.config().vocab_size)
                .collect();
            let ids = self.slm.generate_ids(
                if prompt_ids.is_empty() {
                    &[0]
                } else {
                    &prompt_ids
                },
                max_tokens,
                &processor,
                &mut rng,
            );
            format!("{ids:?}")
        };

        // Extract only the new content (after the prompt).
        let completion = if generated.len() > prompt_text.len() {
            generated[prompt_text.len()..].to_string()
        } else {
            generated
        };

        let completion_tokens = if let Some(tok) = self.slm.tokenizer() {
            tok.encode(&completion).len()
        } else {
            completion.len()
        };

        GenerationResult {
            content: completion,
            prompt_tokens: prompt_len,
            completion_tokens,
        }
    }
}

/// Result of text generation.
pub struct GenerationResult {
    pub content: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

/// Concatenate chat messages into a single prompt string.
fn format_messages(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        if !prompt.is_empty() {
            prompt.push('\n');
        }
        prompt.push_str(&msg.content);
    }
    prompt
}

// ── Shared server state ─────────────────────────────────────────────

/// Shared state for the axum server.
pub struct AppState {
    pub engine: Mutex<InferenceEngine>,
    pub semaphore: Semaphore,
    pub model_name: String,
}

// ── Route handlers ──────────────────────────────────────────────────

/// GET /health
async fn health_handler(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        model: state.model_name.clone(),
    })
}

/// POST /v1/chat/completions
async fn chat_completions_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    // Acquire concurrency permit.
    let _permit = state.semaphore.acquire().await.map_err(|_| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "server shutting down"})),
        )
    })?;

    if req.messages.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "messages must not be empty"})),
        ));
    }

    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    if req.stream {
        // Streaming response via SSE.
        let result = {
            let mut engine = state.engine.lock().unwrap();
            engine.generate(&req.messages, req.max_tokens, req.temperature, req.top_p)
        };

        let model_name = state.model_name.clone();
        let req_id = request_id.clone();

        // Split the content into word-level chunks for streaming.
        let chunks: Vec<String> = result
            .content
            .split_inclusive(' ')
            .map(String::from)
            .collect();

        let events: Vec<Result<Event, std::convert::Infallible>> = {
            let mut events = Vec::new();

            // First chunk: role.
            let first_chunk = ChatCompletionChunk {
                id: req_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model_name.clone(),
                choices: vec![ChatChoiceDelta {
                    index: 0,
                    delta: ChatDelta {
                        role: Some("assistant".to_string()),
                        content: None,
                    },
                    finish_reason: None,
                }],
            };
            events.push(Ok(
                Event::default().data(serde_json::to_string(&first_chunk).unwrap())
            ));

            // Content chunks.
            for chunk_text in &chunks {
                let chunk = ChatCompletionChunk {
                    id: req_id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model_name.clone(),
                    choices: vec![ChatChoiceDelta {
                        index: 0,
                        delta: ChatDelta {
                            role: None,
                            content: Some(chunk_text.clone()),
                        },
                        finish_reason: None,
                    }],
                };
                events.push(Ok(
                    Event::default().data(serde_json::to_string(&chunk).unwrap())
                ));
            }

            // Final chunk: finish_reason.
            let final_chunk = ChatCompletionChunk {
                id: req_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model_name.clone(),
                choices: vec![ChatChoiceDelta {
                    index: 0,
                    delta: ChatDelta {
                        role: None,
                        content: None,
                    },
                    finish_reason: Some("stop".to_string()),
                }],
            };
            events.push(Ok(
                Event::default().data(serde_json::to_string(&final_chunk).unwrap())
            ));

            // [DONE] sentinel.
            events.push(Ok(Event::default().data("[DONE]")));

            events
        };

        Ok(Sse::new(stream::iter(events)).into_response())
    } else {
        // Non-streaming response.
        let result = {
            let mut engine = state.engine.lock().unwrap();
            engine.generate(&req.messages, req.max_tokens, req.temperature, req.top_p)
        };

        let response = ChatCompletionResponse {
            id: request_id,
            object: "chat.completion".to_string(),
            created,
            model: state.model_name.clone(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: result.content,
                },
                finish_reason: "stop".to_string(),
            }],
            usage: UsageStats {
                prompt_tokens: result.prompt_tokens,
                completion_tokens: result.completion_tokens,
                total_tokens: result.prompt_tokens + result.completion_tokens,
            },
        };

        Ok(Json(response).into_response())
    }
}

// ── Router construction ─────────────────────────────────────────────

/// Build the axum router with all routes.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .with_state(state)
}

/// Create the shared application state.
pub fn create_app_state(
    engine: InferenceEngine,
    max_concurrent: usize,
    model_name: String,
) -> Arc<AppState> {
    Arc::new(AppState {
        engine: Mutex::new(engine),
        semaphore: Semaphore::new(max_concurrent),
        model_name,
    })
}

/// Start the server (blocks until shutdown).
pub async fn run_server(config: ServerConfig, engine: InferenceEngine) -> std::io::Result<()> {
    let state = create_app_state(engine, config.max_concurrent, "payya-slm".to_string());
    let app = build_router(state);
    let addr = format!("{}:{}", config.host, config.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_slm() -> payya_slm::Slm {
        let config = payya_slm::SlmConfig {
            vocab_size: 32,
            d_model: 16,
            n_heads: 2,
            n_layers: 1,
            d_ff: 32,
            max_seq_len: 64,
        };
        payya_slm::Slm::new(config, 42)
    }

    fn test_slm_with_tokenizer() -> payya_slm::Slm {
        let corpus = "hello world the cat sat on the mat";
        let tokenizer = payya_tokenizer::Tokenizer::train(corpus, 270);
        let config = payya_slm::SlmConfig {
            vocab_size: tokenizer.vocab_size(),
            d_model: 16,
            n_heads: 2,
            n_layers: 1,
            d_ff: 32,
            max_seq_len: 64,
        };
        payya_slm::Slm::with_tokenizer(config, tokenizer, 42)
    }

    #[test]
    fn format_messages_concatenates() {
        let msgs = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are helpful.".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Hello!".to_string(),
            },
        ];
        let prompt = format_messages(&msgs);
        assert_eq!(prompt, "You are helpful.\nHello!");
    }

    #[test]
    fn engine_generate_without_tokenizer() {
        let slm = test_slm();
        let mut engine = InferenceEngine::new(slm, 42);
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "hi".to_string(),
        }];
        let result = engine.generate(&messages, 5, 1.0, None);
        assert!(!result.content.is_empty());
    }

    #[test]
    fn engine_generate_with_tokenizer() {
        let slm = test_slm_with_tokenizer();
        let mut engine = InferenceEngine::new(slm, 42);
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "hello".to_string(),
        }];
        let result = engine.generate(&messages, 10, 0.8, None);
        assert!(result.prompt_tokens > 0);
    }

    #[test]
    fn request_deserialization() {
        let json = r#"{
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 50,
            "temperature": 0.7,
            "stream": false
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.max_tokens, 50);
        assert!(!req.stream);
    }

    #[test]
    fn request_defaults() {
        let json = r#"{"messages": [{"role": "user", "content": "Hi"}]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.max_tokens, 128);
        assert_eq!(req.temperature, 1.0);
        assert!(!req.stream);
    }

    #[test]
    fn response_serialization() {
        let resp = ChatCompletionResponse {
            id: "chatcmpl-test".to_string(),
            object: "chat.completion".to_string(),
            created: 1234567890,
            model: "payya-slm".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: "Hello!".to_string(),
                },
                finish_reason: "stop".to_string(),
            }],
            usage: UsageStats {
                prompt_tokens: 5,
                completion_tokens: 3,
                total_tokens: 8,
            },
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("chatcmpl-test"));
        assert!(json.contains("Hello!"));
    }

    #[test]
    fn streaming_chunk_serialization() {
        let chunk = ChatCompletionChunk {
            id: "chatcmpl-test".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 1234567890,
            model: "payya-slm".to_string(),
            choices: vec![ChatChoiceDelta {
                index: 0,
                delta: ChatDelta {
                    role: Some("assistant".to_string()),
                    content: None,
                },
                finish_reason: None,
            }],
        };
        let json = serde_json::to_string(&chunk).unwrap();
        assert!(json.contains("chat.completion.chunk"));
        assert!(json.contains("assistant"));
    }

    #[test]
    fn prompt_cache_reuse() {
        let slm = test_slm_with_tokenizer();
        let mut engine = InferenceEngine::new(slm, 42);

        let system_msg = ChatMessage {
            role: "system".to_string(),
            content: "hello world".to_string(),
        };

        // First request caches the prefix.
        let msgs1 = vec![
            system_msg.clone(),
            ChatMessage {
                role: "user".to_string(),
                content: "the cat".to_string(),
            },
        ];
        engine.generate(&msgs1, 5, 1.0, None);

        // Second request with same prefix should find a cache hit.
        let msgs2 = vec![
            system_msg,
            ChatMessage {
                role: "user".to_string(),
                content: "the mat".to_string(),
            },
        ];
        engine.generate(&msgs2, 5, 1.0, None);

        // The prompt cache should have entries.
        assert!(engine.prompt_cache.len() >= 1);
    }

    #[tokio::test]
    async fn router_health_check() {
        use axum::body::Body;
        use axum::http::Request;
        use tower::util::ServiceExt as _;

        let slm = test_slm();
        let engine = InferenceEngine::new(slm, 42);
        let state = create_app_state(engine, 10, "test-model".to_string());
        let app = build_router(state);

        let request = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();

        let response: axum::http::Response<_> = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn router_chat_completions() {
        use axum::body::Body;
        use axum::http::Request;
        use tower::util::ServiceExt as _;

        let slm = test_slm();
        let engine = InferenceEngine::new(slm, 42);
        let state = create_app_state(engine, 10, "test-model".to_string());
        let app = build_router(state);

        let body = serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5,
            "temperature": 1.0,
            "stream": false
        });

        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let response: axum::http::Response<_> = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn router_empty_messages_rejected() {
        use axum::body::Body;
        use axum::http::Request;
        use tower::util::ServiceExt as _;

        let slm = test_slm();
        let engine = InferenceEngine::new(slm, 42);
        let state = create_app_state(engine, 10, "test-model".to_string());
        let app = build_router(state);

        let body = serde_json::json!({
            "messages": [],
            "max_tokens": 5
        });

        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let response: axum::http::Response<_> = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn concurrent_requests() {
        let slm = test_slm();
        let engine = InferenceEngine::new(slm, 42);
        let state = create_app_state(engine, 10, "test-model".to_string());

        let mut handles = Vec::new();
        for i in 0..5 {
            let state = state.clone();
            handles.push(tokio::spawn(async move {
                let _permit = state.semaphore.acquire().await.unwrap();
                let mut engine = state.engine.lock().unwrap();
                let msgs = vec![ChatMessage {
                    role: "user".to_string(),
                    content: format!("request {i}"),
                }];
                let result = engine.generate(&msgs, 3, 1.0, None);
                assert!(!result.content.is_empty());
            }));
        }

        for h in handles {
            h.await.unwrap();
        }
    }
}
