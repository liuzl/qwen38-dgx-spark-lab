#!/usr/bin/env bash
# Run from the pinned Strata source directory. No driver or global package changes.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
export PATH="$PWD/.venv/bin:/usr/local/cuda-12.3/bin:$PATH"
cmake -S . -B build-v100 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_BUILD_TESTS=OFF -DSTRATA_PORTABLE=ON -DGGML_NATIVE=OFF \
  -DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.3/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.3 \
  -DCUDA_CUDART=/usr/local/cuda-12.3/lib64/libcudart.so \
  -DCUDA_cudart_LIBRARY=/usr/local/cuda-12.3/lib64/libcudart.so \
  -DCUDA_cublas_LIBRARY=/usr/local/cuda-12.3/lib64/libcublas.so \
  -DCUDA_cublasLt_LIBRARY=/usr/local/cuda-12.3/lib64/libcublasLt.so \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 \
  -DSTRATA_GGML_DIR="$PWD/third_party/llama.cpp"
cmake --build build-v100 --target strata -j 4
test -x build-v100/strata
file build-v100/strata
deps=$(ldd build-v100/strata)
printf '%s\n' "$deps"
if printf '%s\n' "$deps" | grep -Eq 'not found|libcudart.so.11|libcublas(Lt)?.so.11'; then
  echo 'Invalid CUDA dependency resolution' >&2
  exit 1
fi
/usr/local/cuda-12.3/bin/cuobjdump --list-elf build-v100/strata
sha256sum build-v100/strata
