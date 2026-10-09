# syntax=docker/dockerfile:1
# CPU-only build stage for pinned Strata fb58e0d, x86_64 AVX2 + SM80/SM89.
# Context: clean Strata source plus strata-pilot-build-engine.py copied from
# build-engine.py beside this Dockerfile. See README.md for bounded build commands.
# This intermediate image is not the deployment image: Dockerfile.runtime adds
# qualification tools and bypasses upstream automatic setup/download/rebuild.

FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

# STRATA_EXECV=1: setup.py replaces itself with the server, so the server is PID 1
# and docker stop's SIGTERM reaches it (see setup.start). Normal Linux starts, which
# don't set it, keep spawning the server as a child.
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 LANG=C.UTF-8 STRATA_EXECV=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git libatomic1 libgomp1 \
        python3 python3-pip python3-venv unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/strata
COPY . .

# RTX 20 (75), RTX 30 (86), RTX 40 (89), RTX 50 (120), plus 80 for A-series. CMakeLists
# refuses anything below 75. BUILD_VISION=0 skips the image encoder build.
ARG CUDA_ARCHITECTURES="80;89"
ARG BUILD_VISION=1

RUN python3 -m venv .venv \
    && .venv/bin/pip install --no-cache-dir --upgrade pip \
    && .venv/bin/pip install --no-cache-dir -r requirements.txt \
    && chmod +x setup.sh docker-entrypoint.sh

# llama.cpp at the pinned commit, then the engine and the image encoder, built
# with portable AVX2 CPU code and explicit A100/Ada GPU targets. BUILD.json is what setup.py reads to
# decide whether an engine is current: source=local with a matching src hash
# means the first start reuses it instead of recompiling.
COPY strata-pilot-build-engine.py /opt/strata/build-engine.py
RUN .venv/bin/python build-engine.py \
    && test -x engine/strata \
    && test -x engine/strata-vision \
    && test -s engine/BUILD.json

# the cmake trees are build-time only; the engine itself is what the container needs
RUN rm -rf build build-vision

VOLUME ["/data"]
EXPOSE 8080

# /health is answered before the API key gate, so it works with or without one.
# The port only opens after the model loads (1-3 minutes, longer on a first run),
# so the start period is generous: a too short one marks a still-loading container
# unhealthy and a restart policy would kill it mid-download.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
  CMD curl -fs "http://127.0.0.1:${PORT:-8080}/health" || exit 1

ENTRYPOINT ["./docker-entrypoint.sh"]
