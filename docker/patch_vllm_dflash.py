#!/usr/bin/env python3
"""Apply fail-closed DFlash2 fixes to the pinned vLLM image."""

from __future__ import annotations

import argparse
from pathlib import Path


def replace(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: anchor count {count}, expected 1")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=Path, required=True)
    args = parser.parse_args()

    dflash = args.site / "model_executor/models/qwen3_dflash.py"
    replace(
        dflash,
        "class DFlashQwen3Model(nn.Module):\n    hf_to_vllm_mapper",
        "class DFlashQwen3Model(nn.Module):\n"
        "    # Upstream PR #53449: subclasses such as DFlash2 select their layer.\n"
        "    decoder_layer_cls: type[nn.Module] = DFlashQwen3DecoderLayer\n\n"
        "    hf_to_vllm_mapper",
    )
    replace(
        dflash,
        "                DFlashQwen3DecoderLayer(\n",
        "                self.decoder_layer_cls(\n",
    )

    speculative = args.site / "config/speculative.py"
    anchor = """        factors.append(uses_aux_hidden_states)

        if uses_aux_hidden_states and self.draft_model_config is not None:
"""
    replacement = """        factors.append(uses_aux_hidden_states)

        # Upstream PR #53292: parallel draft depth changes compiled graph shapes.
        if self.method in ("dflash", "dspark"):
            factors.append(self.num_speculative_tokens)

        if uses_aux_hidden_states and self.draft_model_config is not None:
"""
    replace(speculative, anchor, replacement)

    print("Applied DFlash2 loader fix (#53449) and K-specific cache key (#53292)")


if __name__ == "__main__":
    main()
