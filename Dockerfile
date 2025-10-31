# --- Stage 1: The "Builder" ---
FROM rust:1.80-bookworm AS builder

# Install build prerequisites...
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake clang libclang-dev pkg-config libssl-dev \
    libopencv-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    curl ca-certificates tar

# Update Rust toolchain to nightly first
RUN rustup update nightly && rustup default nightly

# Download and Extract ONNX Runtime...
ARG ORT_VERSION=1.22.0 
# Using the version confirmed to download
ARG ORT_RELEASE_URL=https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz
RUN curl -sSL -o /tmp/onnxruntime.tgz ${ORT_RELEASE_URL} \
    && file /tmp/onnxruntime.tgz | grep -q 'gzip compressed data' \
    && mkdir -p /opt/onnxruntime \
    && tar -xzf /tmp/onnxruntime.tgz -C /opt/onnxruntime --strip-components=1 \
    && rm /tmp/onnxruntime.tgz

WORKDIR /app

# Copy ONLY Cargo.toml first - allows Docker to cache dependency downloads/builds
COPY Cargo.toml ./
# Copy source code AFTER manifests
COPY src ./src

# Set ENVs needed for build
ENV ORT_LIB_LOCATION=/opt/onnxruntime
ENV OPENCV_BUILD_FROM_SOURCE=1
ENV OPENCV_BUILD_CONTRIB=1

# Clean potential old artifacts AND build using the nightly toolchain
RUN cargo clean && cargo build --release

# --- Stage 2: The "Final App" ---
FROM debian:bookworm-slim
ARG ORT_VERSION=1.22.0 
# Must match version from Stage 1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-core406 libopencv-imgproc406 libopencv-videoio406 libopencv-highgui406 \
    libavcodec59 libavformat59 libswscale6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/target/release/video-reconstruction-from-jumbled-frames .
COPY --from=builder /opt/onnxruntime/lib /usr/local/lib
RUN ldconfig
COPY resnet50.onnx .

COPY resnet50.onnx.data .
RUN mkdir -p /app/test_videos /app/output_videos
ENV ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so.${ORT_VERSION}
CMD ["./video-reconstruction-from-jumbled-frames"]