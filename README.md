# inferstack-rs

`inferstack-rs` is a lightweight Rust ONNX inference service built around Axum and Tract. It supports schema-aware model loading, version-based routing, batch execution, optional Redis caching, and Prometheus/structured-log observability.

The service is designed as a general ONNX inference microservice, not a full ML pipeline platform. It focuses on serving already-prepared tensors and returning model outputs, while leaving most preprocessing/postprocessing to clients or upstream services.

## Features

- ONNX model serving via the Tract runtime
- Schema-aware model introspection (input/output names, shapes, dtypes) at load time
- Named input and output tensor support in API responses
- Batch inference execution (`/batch`) with per-item output mapping
- Deterministic weighted version routing (traffic allocations summing to 100)
- Explicit model version override per request (`model_version`)
- Optional Redis-based caching for inference outputs
- Prometheus metrics endpoint (`/metrics`)
- Structured tracing logs with request IDs
- Docker Compose deployment with Redis, Prometheus, and Grafana
- Health and metadata endpoints (`/health`, `/models`)
- Test coverage for routing, inference, validation, and caching behavior

## Architecture

```text
Client
  -> HTTP API (Axum)
     -> request parsing + validation
     -> model routing (explicit version or weighted selection)
     -> inference engine (single or batch)
     -> ONNX runtime (Tract)
     -> optional Redis cache read/write
  -> response mapping (legacy output + named outputs)

Operational endpoints:
  /health, /models, /metrics
```

Layer responsibilities:

- HTTP/API layer: receives JSON requests, validates payload shape/size, and exposes operational endpoints.
- Routing layer: chooses a model version using either `model_version` override or weighted routing table.
- Inference layer: handles single and batch execution, output name mapping, and fallback behavior for some batch execution failures.
- Runtime layer: executes ONNX graphs via Tract.
- Cache layer (optional): caches per-input inference outputs in Redis with configurable TTL behavior.
- Observability layer: emits Prometheus counters/histograms and request-scoped tracing logs.

## API Usage

### Single inference

Endpoint:

```http
POST /infer
Content-Type: application/json
```

Legacy input format (currently the most permissive path):

```json
{
  "input": [[1.0, 2.0, 3.0, 4.0]]
}
```

Named input format (currently supported for non-batch requests and models with exactly one input tensor):

```json
{
  "inputs": {
    "input_tensor": [1.0, 2.0, 3.0, 4.0]
  }
}
```

Example response:

```json
{
  "output": [[1.0, 2.0, 3.0, 4.0]],
  "outputs": {
    "output_tensor": [1.0, 2.0, 3.0, 4.0]
  }
}
```

### Batch inference

Endpoint:

```http
POST /batch
Content-Type: application/json
```

Example request:

```json
{
  "input": [
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0]
  ]
}
```

Notes:

- `/batch` forces batch mode server-side.
- Named `inputs` is not currently accepted in batch mode.

### Model metadata

Endpoint:

```http
GET /models
```

Returns model versions and each version's input/output schema (`name`, `shape`, `dtype`).

## Observability

- `/health`: readiness/status (`models_loaded`, `redis_connected`)
- `/metrics`: Prometheus text-format metrics (request counts, errors, durations, cache hit/miss counters)
- `/models`: runtime model metadata and schema introspection output

## Supported Model Types

Model categories that generally fit this service well:

- tabular models
- embedding models
- transformer encoder-style models
- simple CNN classifiers
- ONNX models with tensor inputs/outputs that can be sent directly as numeric arrays

## Limitations

- No built-in preprocessing/postprocessing pipelines
- Image/audio tokenization, normalization, decoding, etc. must be handled by clients/upstream services
- Detection/segmentation models (for example YOLO variants) need external postprocessing (NMS, label mapping, box decoding)
- Runtime execution path currently expects float32 tensor inputs
- Current request validation and execution path is centered on single-input models
- Named `inputs` payload is currently non-batch only
- Batch mode requires consistent per-item tensor lengths and respects `MAX_BATCH_SIZE`
- Dynamic dimensions are partially handled, but practical compatibility depends on model graph/runtime constraints

## Running the Service

### Local

```bash
cargo run
```

### Docker Compose

```bash
docker-compose up
```

The compose stack starts:

- app service
- Redis
- Redis exporter
- Prometheus
- Grafana

### Configuration

Configuration is environment-variable driven.

Core variables:

- `MODEL_PATH` (default: `model.onnx`) - used when `MODEL_VERSIONS` is not set
- `MODEL_VERSIONS` - comma-separated entries in format `<version>:<path>:<allocation>`
- `DEFAULT_MODEL_VERSION` - optional explicit default
- `PORT` (default: `3000`)
- `REDIS_URL` - optional, enables cache when set
- `CACHE_TTL` - optional cache TTL in seconds
- `REDIS_POOL_SIZE` (default: `5`)
- `MIN_INFERENCE_MS_FOR_CACHE` (default: `5`)
- `MAX_BATCH_SIZE` (default: `32`)
- `MIN_INPUT_SIZE` / `MAX_INPUT_SIZE`
- `NORMALIZE_INPUT` (`none` or `minmax`)
- `RATE_LIMIT_REQUESTS`, `RATE_LIMIT_WINDOW_SECS`, `RATE_LIMIT_CLEANUP_INTERVAL`

Model configuration requirements:

- at least one model must be configured
- traffic allocations across versions must sum to 100
- each configured model path must exist at startup

## Testing

Run the test suite:

```bash
cargo test
```

Notes:

- Integration tests use ONNX fixtures under `tests/fixtures`.
- Cache integration tests include Redis-backed paths; one cache test is marked ignored and requires Docker/testcontainers or `REDIS_TEST_URL`.

## Benchmarking

No dedicated benchmark scripts or `benches/` targets are currently included in this repository.

## Design Goals

- demonstrate a pragmatic Rust ONNX serving architecture
- explore weighted model routing, batching, and cache-assisted inference
- provide a minimal, inspectable inference microservice with operational endpoints and tests

This repository is best treated as a focused serving service and reference implementation, not a full production ML platform.