#!/usr/bin/env python3
"""Convert Qwen3.8 output-space refusal directions into a rank-1 PEFT LoRA.

The source hook applies, for each affected linear output::

    y <- y - lambda * coef * r * (r.T @ y)

For the linear weight W this is the rank-1 update::

    delta_W = (-coef * r) @ (r.T @ W)

The generated adapter fixes lambda=1. Base requests select no adapter; the
modified alias selects this adapter through vLLM's native per-request LoRA.

RadixArk Qwen3.8 uses static FP8 for attention/GDN output projections and
packed NVFP4 for MLP down projections. The official FP8 checkpoint uses E4M3
weights with two-dimensional 128x128 inverse block scales. The converter also
supports checkpoints without a safetensors index by discovering layer shards.
It consumes all three layouts directly and never materializes a complete
dequantized model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP4_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unpack_nvfp4(packed: torch.Tensor) -> torch.Tensor:
    """Decode packed E2M1 bytes to float32, low nibble then high nibble."""
    if packed.dtype != torch.uint8:
        raise TypeError(f"NVFP4 weight must be uint8, got {packed.dtype}")
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    nibbles = torch.stack((low, high), dim=-1).reshape(packed.shape[0], -1)
    magnitude = nibbles & 0x07
    sign = torch.where((nibbles & 0x08) != 0, -1.0, 1.0)
    return FP4_E2M1[magnitude.long()] * sign


def fp8_weight_chunk(
    weight: torch.Tensor,
    scale: torch.Tensor,
    start: int,
    end: int,
    block_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    chunk = weight[start:end].float()
    scale_f32 = scale.float()
    if scale_f32.numel() == 1:
        return chunk * scale_f32
    if scale_f32.ndim == 1 and scale_f32.shape[0] == weight.shape[0]:
        return chunk * scale_f32[start:end, None]
    if scale_f32.ndim == 2 and block_size is not None:
        row_block, column_block = block_size
        expected = (
            (weight.shape[0] + row_block - 1) // row_block,
            (weight.shape[1] + column_block - 1) // column_block,
        )
        if tuple(scale_f32.shape) != expected:
            raise ValueError(
                "FP8 block scale layout mismatch: "
                f"weight={tuple(weight.shape)} scale={tuple(scale.shape)} "
                f"expected={expected}"
            )
        row_indices = torch.arange(start, end) // row_block
        column_indices = torch.arange(weight.shape[1]) // column_block
        expanded = scale_f32[row_indices][:, column_indices]
        return chunk * expanded
    raise ValueError(f"unsupported FP8 weight_scale shape {tuple(scale.shape)}")


def nvfp4_weight_chunk(
    weight: torch.Tensor,
    block_scale: torch.Tensor,
    global_scale: torch.Tensor,
    start: int,
    end: int,
) -> torch.Tensor:
    packed = weight[start:end]
    decoded = unpack_nvfp4(packed).reshape(end - start, -1, 16)
    scales = block_scale[start:end].float().unsqueeze(-1)
    if decoded.shape[:2] != scales.shape[:2]:
        raise ValueError(
            "NVFP4 scale layout mismatch: "
            f"decoded={tuple(decoded.shape)} scales={tuple(scales.shape)}"
        )
    return (decoded * scales * global_scale.float()).reshape(end - start, -1)


def adapter_module_name(direction_name: str) -> str:
    prefix = "model.language_model."
    if not direction_name.startswith(prefix):
        raise ValueError(f"unexpected direction module name: {direction_name}")
    return "language_model.model." + direction_name[len(prefix) :]


def build_lora_pair(
    shard: Path,
    module: str,
    direction: torch.Tensor,
    coef: float,
    row_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    weight_name = f"{module}.weight"
    with safe_open(shard, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(weight_name)
        if weight.shape[0] != direction.numel():
            raise ValueError(
                f"{module}: output mismatch weight={tuple(weight.shape)} "
                f"direction={tuple(direction.shape)}"
            )

        if weight.dtype == torch.uint8:
            block_scale = handle.get_tensor(f"{module}.weight_scale")
            global_scale = handle.get_tensor(f"{module}.weight_scale_2")
            input_size = weight.shape[1] * 2
            quant = "nvfp4"

            def get_chunk(start: int, end: int) -> torch.Tensor:
                return nvfp4_weight_chunk(weight, block_scale, global_scale, start, end)

        elif weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            keys = set(handle.keys())
            inverse_scale_name = f"{module}.weight_scale_inv"
            scale_name = f"{module}.weight_scale"
            if inverse_scale_name in keys:
                scale = handle.get_tensor(inverse_scale_name)
                block_size = (128, 128)
                quant = "fp8_block_128x128"
            elif scale_name in keys:
                scale = handle.get_tensor(scale_name)
                block_size = None
                quant = "fp8"
            else:
                raise KeyError(f"FP8 scale not found for {module}")
            input_size = weight.shape[1]

            def get_chunk(start: int, end: int) -> torch.Tensor:
                return fp8_weight_chunk(weight, scale, start, end, block_size)

        elif weight.dtype in (torch.float16, torch.bfloat16, torch.float32):
            input_size = weight.shape[1]
            quant = str(weight.dtype).removeprefix("torch.")

            def get_chunk(start: int, end: int) -> torch.Tensor:
                return weight[start:end].float()

        else:
            raise TypeError(f"{module}: unsupported weight dtype {weight.dtype}")

        direction = direction.float()
        lora_a = torch.zeros(input_size, dtype=torch.float32)
        for start in range(0, weight.shape[0], row_chunk):
            end = min(start + row_chunk, weight.shape[0])
            effective_weight = get_chunk(start, end)
            lora_a.add_(torch.sum(direction[start:end, None] * effective_weight, dim=0))

    lora_b = -float(coef) * direction
    return lora_a.unsqueeze(0), lora_b.unsqueeze(1), quant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row-chunk", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def discover_weight_map(model: Path) -> tuple[dict[str, str], str, str | None]:
    index_path = model / "model.safetensors.index.json"
    if index_path.exists():
        index: dict[str, Any] = json.loads(index_path.read_text())
        weight_map: dict[str, str] = index["weight_map"]
        return weight_map, "model.safetensors.index.json", sha256(index_path)

    weight_map = {}
    shards = sorted(model.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors shards found under {model}")
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():  # noqa: SIM118 - safe_open is not iterable
                if not key.endswith(".weight"):
                    continue
                if key in weight_map:
                    raise ValueError(f"duplicate weight across shards: {key}")
                weight_map[key] = shard.name
    return weight_map, "safetensors_scan", None


def mapping_sha256(weight_map: dict[str, str]) -> str:
    rendered = json.dumps(weight_map, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode()).hexdigest()


def main() -> None:
    args = parse_args()
    if args.row_chunk <= 0:
        raise SystemExit("--row-chunk must be positive")

    weight_map, weight_map_source, index_sha256 = discover_weight_map(args.model)

    with safe_open(args.directions, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        module_order = json.loads(metadata["coef_order"])
        coefs = handle.get_tensor("__coefs__").float()
        directions = {name: handle.get_tensor(name).float() for name in module_order}

    if len(module_order) != len(coefs):
        raise ValueError(
            f"direction/coef mismatch: {len(module_order)} != {len(coefs)}"
        )

    if args.output.exists() and any(args.output.iterdir()) and not args.force:
        raise SystemExit(f"output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    adapter: dict[str, torch.Tensor] = {}
    quant_counts: dict[str, int] = {}
    module_manifest: list[dict[str, Any]] = []

    for idx, module in enumerate(module_order):
        weight_name = f"{module}.weight"
        shard_name = weight_map.get(weight_name)
        if shard_name is None:
            raise KeyError(f"weight not found in index: {weight_name}")
        lora_a, lora_b, quant = build_lora_pair(
            args.model / shard_name,
            module,
            directions[module],
            float(coefs[idx]),
            args.row_chunk,
        )
        vllm_name = adapter_module_name(module)
        key_prefix = f"base_model.model.{vllm_name}"
        adapter[f"{key_prefix}.lora_A.weight"] = lora_a.contiguous()
        adapter[f"{key_prefix}.lora_B.weight"] = lora_b.contiguous()
        quant_counts[quant] = quant_counts.get(quant, 0) + 1
        module_manifest.append(
            {
                "source_module": module,
                "adapter_module": vllm_name,
                "quantization": quant,
                "input_size": lora_a.shape[1],
                "output_size": lora_b.shape[0],
                "coef": float(coefs[idx]),
            }
        )
        print(
            f"[{idx + 1:03d}/{len(module_order)}] {module} "
            f"{quant} {tuple(lora_b.shape)}x{tuple(lora_a.shape)}",
            flush=True,
        )

    adapter_path = args.output / "adapter_model.safetensors"
    save_file(
        adapter,
        adapter_path,
        metadata={
            "format": "pt",
            "conversion": "qwen38-output-rank1-to-lora",
            "directions_sha256": sha256(args.directions),
        },
    )

    adapter_config = {
        "base_model_name_or_path": str(args.model),
        "bias": "none",
        "inference_mode": True,
        "lora_alpha": 1,
        "lora_dropout": 0.0,
        "peft_type": "LORA",
        "r": 1,
        "target_modules": ["out_proj", "o_proj", "down_proj"],
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }
    (args.output / "adapter_config.json").write_text(
        json.dumps(adapter_config, indent=2, sort_keys=True) + "\n"
    )

    manifest = {
        "model": str(args.model),
        "model_index_sha256": index_sha256,
        "model_weight_map_source": weight_map_source,
        "model_weight_map_sha256": mapping_sha256(weight_map),
        "directions": str(args.directions),
        "directions_sha256": sha256(args.directions),
        "adapter_sha256": sha256(adapter_path),
        "modules": len(module_manifest),
        "quantization_counts": quant_counts,
        "module_manifest": module_manifest,
    }
    (args.output / "conversion-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in ("adapter_sha256", "modules", "quantization_counts")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
