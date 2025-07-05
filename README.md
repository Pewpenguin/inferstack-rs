# InferStack-rs

A high-performance, production-ready inference server written in Rust that supports model versioning, A/B testing, caching, and comprehensive monitoring.

## Features

- **Model Versioning**: Support multiple model versions with traffic allocation
- **A/B Testing**: Configure traffic distribution across different model versions
- **Caching**: Redis-based caching for inference results
- **Rate Limiting**: Configurable rate limiting per client
- **Monitoring**: Comprehensive metrics via Prometheus and Grafana dashboards
- **Batch Processing**: Efficient handling of batch inference requests
- **Input Validation**: Configurable input size limits and validation
- **Health Checks**: Built-in health monitoring endpoints
- **Graceful Shutdown**: Proper shutdown handling with cleanup

## Quick Start

### Prerequisites

- Rust (latest stable version)
- Redis (optional, for caching)
- Docker and Docker Compose (optional, for containerization)

### Installation

```bash
# Clone the repository
git clone https://github.com/Pewpenguin/inferstack-rs
cd inferstack-rs

# Build the project
cargo build --release
```

### Configuration

Set up the server using environment variables defined in a `.env` file.

Use the `.env.example` file as a reference for the required structure and variable names. Ensure all necessary variables are defined before starting the server.

### Running

```bash
# Run directly
cargo run --release

# Or using Docker
docker-compose up -d
```

## API Endpoints

### Health Check
```http
GET /health
```

### Inference
```http
POST /inference
Content-Type: application/json

{
  "input": [[1.0, 2.0, 3.0]],
  "model_version": "v1"  // optional
}
```

### Batch Inference
```http
POST /inference
Content-Type: application/json

{
  "input": [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
  ],
  "batch": true,
  "model_version": "v1"  // optional
}
```

## Monitoring

Access metrics at `/metrics` endpoint. Key metrics include:

- `inferstack_inference_total`: Total inference requests
- `inferstack_model_version_usage_total`: Usage by model version
- `inferstack_inference_duration_seconds`: Inference latency
- `inferstack_cache_operations_total`: Cache operation statistics
- `inferstack_batch_throughput_items_per_second`: Batch processing performance

### Grafana Dashboard

A pre-configured Grafana dashboard is available in `monitoring/grafana/dashboards/`.

## Development

### Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.