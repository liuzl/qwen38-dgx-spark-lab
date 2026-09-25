#!/usr/bin/env python3
"""Apply the remaining fail-closed DFlash2 fix to vLLM v0.28.0.

#53449 (subclass-selectable decoder layer) is upstream since v0.28.0.
#53292 (parallel draft depth in the compile-cache key) is still missing through
v0.30.0; on v0.28.0 it applies at the same anchor as patch_vllm_dflash.py.
"""

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
    if "decoder_layer_cls = DFlashQwen3DecoderLayer" not in dflash.read_text(encoding="utf-8"):
        raise SystemExit(f"{dflash}: expected upstream #53449 decoder_layer_cls")

    speculative = args.site / "config/speculative.py"
    replace(
        speculative,
        "        factors.append(uses_aux_hidden_states)\n\n"
        "        if uses_aux_hidden_states and self.draft_model_config is not None:\n",
        "        factors.append(uses_aux_hidden_states)\n\n"
        "        # Upstream PR #53292: parallel draft depth changes compiled graph shapes.\n"
        "        if self.method in (\"dflash\", \"dspark\"):\n"
        "            factors.append(self.num_speculative_tokens)\n\n"
        "        if uses_aux_hidden_states and self.draft_model_config is not None:\n",
    )
    print("Verified #53449 upstream; applied K-specific cache key (#53292)")


if __name__ == "__main__":
    main()
