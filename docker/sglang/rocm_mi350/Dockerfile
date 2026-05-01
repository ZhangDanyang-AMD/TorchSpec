ARG BASE_IMAGE=registry-sc-harbor.amd.com/framework/therock-main:1239_gfx950_7.13.0a20260413_ubuntu24.04_py3.12_pytorch_release-2.11_0446f7b
FROM ${BASE_IMAGE}

WORKDIR /root/

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git rsync dnsutils libibverbs-dev libnuma-dev \
        cmake build-essential libjsoncpp-dev libgoogle-glog-dev \
        wget golang-go pybind11-dev && \
    rm -rf /var/lib/apt/lists/*

# Save ROCm torch wheel location before SGLang clobbers it
RUN pip show torch | grep Location > /tmp/torch_location.txt && \
    pip freeze | grep "^torch==" > /tmp/rocm_torch_version.txt

# Build mooncake transfer engine from source with HIP (ROCm) backend
ARG MOONCAKE_COMMIT=b6a841dc78c707ec655a563453277d969fb8f38d
RUN git clone https://github.com/kvcache-ai/Mooncake.git /tmp/mooncake && \
    cd /tmp/mooncake && \
    git checkout ${MOONCAKE_COMMIT} && \
    git submodule update --init --recursive && \
    bash dependencies.sh -y && \
    mkdir build && cd build && \
    cmake .. -DUSE_HIP=ON -DUSE_ETCD=ON && \
    make -j$(nproc) && make install && \
    cd / && rm -rf /tmp/mooncake

# Install mooncake pip package for Python bindings, then source-built .so takes precedence via LD path
RUN pip install --no-cache-dir "mooncake-transfer-engine==0.3.10.post1" 2>/dev/null || \
    pip install --no-cache-dir "mooncake-transfer-engine-non-cuda>=0.3.10.post1"

# Install SGLang from specific commit + apply TorchSpec patch
ARG SGLANG_COMMIT=94f03a39dbd39edfc2b118b5357bbbadaaa9ad28
RUN git clone --depth=1 --branch main https://github.com/sgl-project/sglang.git /sgl-workspace/sglang && \
    cd /sgl-workspace/sglang && \
    git fetch --depth=1 origin $SGLANG_COMMIT && \
    git checkout $SGLANG_COMMIT

COPY patches/sglang/v0.5.10.post1/sglang.patch /sgl-workspace/
RUN cd /sgl-workspace/sglang && \
    git apply /sgl-workspace/sglang.patch && \
    rm /sgl-workspace/sglang.patch && \
    cd python && pip install --no-deps -e .

# Build sgl-kernel from source for ROCm (gfx950 = MI350)
ARG GPU_ARCH=gfx950
RUN cd /sgl-workspace/sglang/sgl-kernel && \
    PYTORCH_ROCM_ARCH=${GPU_ARCH} GPU_ARCH=${GPU_ARCH} \
    python3 setup_rocm.py install

# Install SGLang runtime deps (excluding torch/triton which are already present via ROCm)
RUN pip install \
    fastapi uvicorn uvloop starlette requests aiohttp \
    openai tiktoken orjson msgspec pillow \
    pydantic psutil setproctitle pybase64 python-multipart \
    nvidia-ml-py prometheus-client py-spy interegular \
    outlines-core outlines lark partial-json-parser compressed-tensors \
    torchao einops sentencepiece modelscope \
    flashinfer-python xgrammar watchfiles \
    openai-harmony torch-memory-saver \
    ipython \
    2>/dev/null || true

# Install TorchSpec (without pulling torch)
COPY . /root/torchspec
RUN cd /root/torchspec && pip install --no-deps -e . && \
    pip install omegaconf ray accelerate wandb pyzmq qwen-vl-utils \
    datasets transformers huggingface_hub safetensors pydantic openai numba

ENV PYTORCH_ROCM_ARCH=gfx950
ENV HIP_FORCE_DEV_KERNARG=1
ENV SGLANG_DISABLE_CUDNN_CHECK=1
ENV SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

WORKDIR /root/torchspec
