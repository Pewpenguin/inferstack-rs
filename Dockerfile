FROM rust:1.87-slim-bookworm as builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y pkg-config libssl-dev && \
    rm -rf /var/lib/apt/lists/*

RUN USER=root cargo new --bin inferstack-rs
WORKDIR /app/inferstack-rs

COPY Cargo.toml Cargo.lock ./

RUN cargo build --release
RUN rm src/*.rs

COPY src ./src

RUN cargo build --release

FROM debian:bookworm-slim	

WORKDIR /app

RUN apt-get update && \
    apt-get install -y ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/inferstack-rs/target/release/inferstack-rs /app/

RUN mkdir -p /app/model

ENV MODEL_PATH=/app/model.onnx
ENV PORT=3000

EXPOSE 3000

CMD ["/app/inferstack-rs"]