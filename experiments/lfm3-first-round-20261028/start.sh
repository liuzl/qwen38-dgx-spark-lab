#!/bin/bash
set -euo pipefail
umask 077
v=$1
cd /workspace/lfm3-eval
nohup python -m sglang.launch_server --model-path /workspace/lfm3-eval/model-$v --served-model-name lfm3 --enable-cache-report --trust-remote-code --reasoning-parser auto --tool-call-parser auto --host 127.0.0.1 --port 30000 --context-length 16384 --max-running-requests 4 --mem-fraction-static 0.3 --cuda-graph-max-bs-decode 4 --max-total-tokens 65536 --max-mamba-cache-size 60762 > engine-$v.log 2>&1 < /dev/null &
echo $! > engine.pid
