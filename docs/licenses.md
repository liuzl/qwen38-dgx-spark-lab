# Model and artifact licenses

Repository-authored code is released under Apache License 2.0. This does not
change the license of anything downloaded or generated separately.

At the time of the reference run:

| Dependency/artifact | Reported license | Distributed here? |
|---|---|---:|
| vLLM | Apache-2.0 | no; referenced as a base image |
| z-lab/dflash code | MIT | no |
| Qwen/Qwen3.8-27B | Apache-2.0 | no |
| RadixArk/Qwen3.8-27B-NVFP4 | Apache-2.0 | no |
| z-lab Qwen3.8 DFlash2 checkpoint | model repository terms | no |
| Ektome Qwen3.8 uncensored checkpoint | Apache-2.0 | no |
| refusal direction tensor | artifact-specific terms | **no** |

Always re-check the model card and LICENSE at the exact revision you download.
The converter output is derived from both the base checkpoint and direction
tensor; responsibility for determining whether that output may be redistributed
rests with the person generating it.

This repository publishes neither generated adapters nor direction vectors.
