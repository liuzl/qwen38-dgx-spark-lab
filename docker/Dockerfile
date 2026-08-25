ARG BASE=vllm/vllm-openai@sha256:4db6d42b66ad393faa3da7341db580f443b7aeb9a7de5597cd11b724eabff6f6
FROM ${BASE}

ARG SITE=/usr/local/lib/python3.12/dist-packages/vllm
COPY patch_vllm_dflash.py /opt/qwen38/patch_vllm_dflash.py
RUN python3 /opt/qwen38/patch_vllm_dflash.py --site "${SITE}" \
    && python3 -m py_compile \
       "${SITE}/model_executor/models/qwen3_dflash.py" \
       "${SITE}/config/speculative.py"

LABEL org.opencontainers.image.title="qwen38-vllm-dflash2-lab" \
      org.opencontainers.image.description="Pinned vLLM DFlash2 overlay for DGX Spark"

