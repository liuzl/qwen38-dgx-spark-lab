from huggingface_hub import snapshot_download
import json,pathlib,time
p=pathlib.Path('/workspace/lfm3-eval')
for v,rev in [('bf16','40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd'),('fp8','243955da95b1cd68026f293b29de4b4c13f3638f')]:
 print('download',v,time.time(),flush=True)
 snapshot_download('LiquidAI/LFM3-1.7B'+('-FP8' if v=='fp8' else ''),revision=rev,local_dir=str(p/('model-'+v)),ignore_patterns=['assets/*'],max_workers=8)
 print('ready',v,time.time(),flush=True)
