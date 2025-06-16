# --- Stage 1: Build ---
FROM rust:1.87-slim-bookworm as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y pkg-config libssl-dev && \
    rm -rf /var/lib/apt/lists/*

# Create a dummy project to cache dependencies
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release && rm -rf src

# Now copy actual source code
COPY src ./src
RUN cargo build --release

# --- Stage 2: Runtime ---
FROM debian:bookworm-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy binary from builder
COPY --from=builder /app/target/release/inferstack-rs /app/inferstack-rs

# Add model directory
RUN mkdir -p /app/model

ENV MODEL_PATH=/app/model.onnx
ENV PORT=3000

EXPOSE 3000

CMD ["/app/inferstack-rs"]