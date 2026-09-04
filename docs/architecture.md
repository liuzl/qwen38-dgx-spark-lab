# Architecture

## Serving topology

One vLLM process loads:

1. a mixed-NVFP4 Qwen3.8-27B target;
2. a separate DFlash2 draft checkpoint;
3. a rank-1 PEFT LoRA adapter registered as a second model ID.

Base requests carry no LoRA ID. Adapter requests use vLLM's native per-request
LoRA mapping. vLLM includes the LoRA identity in the prefix-cache key, which
prevents base and modified activations from sharing KV blocks.

## From output projection to LoRA

The original runtime hook applies this update after a linear sublayer:

```text
y' = y - lambda * coef * r * (r^T y)
```

For `y = W x`, the fixed `lambda=1` update can be written as:

```text
W'     = W - coef * r * (r^T W)
deltaW = B A
B      = -coef * r
A      = r^T W
```

That is a rank-1 LoRA. The converter emits one `A` and `B` pair for each edited
module.

The reference direction set edits 128 residual-writing modules:

- 48 Gated DeltaNet `linear_attn.out_proj` modules;
- 16 full-attention `self_attn.o_proj` modules;
- 64 MLP `down_proj` modules.

## Quantized weight handling

The RadixArk checkpoint is mixed precision rather than uniformly 4-bit. In the
measured revision its three shards contained approximately:

| Stored dtype | GiB |
|---|---:|
| packed NVFP4 (`U8`) | 8.56 |
| FP8 | 7.79 |
| BF16 | 4.07 |
| total | 20.42 |

The converter reads static-FP8 output weights directly. For packed NVFP4 it
decodes E2M1 nibbles, applies per-block FP8 scales and the global scale. For the
official FP8 checkpoint it expands E4M3 `weight_scale_inv` values over their
128x128 blocks and can discover weights from layer shards when no safetensors
index exists. Every path accumulates `r^T W` in row chunks so a full
dequantized model is never resident.

## Important approximation

Post-linear projection and weight-space LoRA are identical for an ordinary
linear map. Quantized activation kernels add rounding, so LoRA applied to the
original input is not assumed to be bit-identical to projecting the quantized
output. This is why safety/behavior qualification is required even when the
algebra and weight conversion are correct.

## Memory

The measured Qwen vLLM process used about 46 GiB:

```text
target checkpoint       20.42 GiB
DFlash2 drafter          3.58 GiB
FP8 KV cache             16.0 GiB
CUDA graphs              2.06 GiB
runtime workspaces       ~4 GiB
```

The adapter itself was 8.3 MiB. Do not run two complete Qwen serving engines
simultaneously merely to expose two behavioral modes; the duplicated weights,
KV pools, and graphs erase the advantage of native LoRA.
