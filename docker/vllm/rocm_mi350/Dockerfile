FROM vllm/vllm-openai-rocm:v0.19.1

WORKDIR /root/

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git rsync dnsutils libibverbs-dev libnuma-dev \
        cmake build-essential libjsoncpp-dev libgoogle-glog-dev \
        wget golang-go pybind11-dev && \
    rm -rf /var/lib/apt/lists/*

# Install mooncake pip package first (provides Python scaffolding + Go mooncake_master binary)
RUN pip install --no-cache-dir "mooncake-transfer-engine==0.3.10.post1" 2>/dev/null || \
    pip install --no-cache-dir "mooncake-transfer-engine-non-cuda>=0.3.10.post1"

# Build mooncake transfer engine from source with HIP backend, then replace pip's .so files
ARG MOONCAKE_COMMIT=f2853a80a494b999d2115449dfd153dd25f5b42d
RUN git clone https://github.com/kvcache-ai/Mooncake.git /tmp/mooncake && \
    cd /tmp/mooncake && \
    git checkout ${MOONCAKE_COMMIT} && \
    git submodule update --init --recursive && \
    bash dependencies.sh -y && \
    pip install pybind11 && \
    export PATH="/usr/local/go/bin:$PATH" && \
    mkdir build && cd build && \
    cmake .. -DUSE_HIP=ON -DUSE_ETCD=ON && \
    make -j$(nproc) && \
    make install && \
    SITE=$(python3 -c "import mooncake; import os; print(os.path.dirname(mooncake.__file__))") && \
    echo "Replacing pip mooncake .so files in $SITE with source-built versions" && \
    SRC_STORE=$(find /tmp/mooncake/build -name 'store*.so' -path '*/mooncake-integration/*' | head -1) && \
    SRC_ENGINE=$(find /tmp/mooncake/build -name 'engine*.so' -path '*/mooncake-integration/*' | head -1) && \
    if [ -n "$SRC_STORE" ]; then \
        rm -f "$SITE"/store*.so && \
        cp -v "$SRC_STORE" "$SITE/store.so" && \
        echo "Replaced all store .so with source-built HIP version"; \
    fi && \
    if [ -n "$SRC_ENGINE" ]; then \
        rm -f "$SITE"/engine*.so && \
        cp -v "$SRC_ENGINE" "$SITE/engine.so" && \
        echo "Replaced all engine .so with source-built HIP version"; \
    fi && \
    SRC_MASTER=$(find /tmp/mooncake/build -name 'mooncake_master' -type f -executable | head -1) && \
    if [ -n "$SRC_MASTER" ]; then cp -v "$SRC_MASTER" "$SITE/mooncake_master"; fi && \
    rm -rf /tmp/mooncake

# Apply TorchSpec patches to vLLM
COPY patches/vllm/v0.19.1/*.patch /tmp/patches/
RUN VLLM_DIR=$(python3 -c "import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))") && \
    cd "$VLLM_DIR" && \
    for p in /tmp/patches/*.patch; do patch -p1 < "$p"; done && \
    rm -rf /tmp/patches

# Install TorchSpec (--no-deps to avoid pulling CUDA mooncake)
COPY . /root/torchspec
RUN cd /root/torchspec && pip install --no-deps -e . && \
    pip install --no-cache-dir omegaconf ray accelerate wandb pyzmq qwen-vl-utils \
    datasets transformers huggingface_hub safetensors pydantic openai numba

ENV PYTORCH_ROCM_ARCH=gfx950
ENV HIP_FORCE_DEV_KERNARG=1

ENTRYPOINT []

WORKDIR /root/torchspec
