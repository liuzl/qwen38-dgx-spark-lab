#!/usr/bin/env bash
# $1 is the dedicated experiment root; no global network changes.
set -euo pipefail
cd "${1:?experiment root required}"
mkdir -p models evidence
printf 'running\n' > evidence/download.status
trap 'printf "failed-or-interrupted\n" > evidence/download.status' EXIT
aria2c --input-file=evidence/strata-download.aria2 --dir=models \
  --continue=true --check-integrity=true --file-allocation=none \
  --max-concurrent-downloads=2 --split=8 --max-connection-per-server=8 \
  --max-tries=12 --retry-wait=15 \
  --connect-timeout=30 --timeout=60 --summary-interval=300 \
  --console-log-level=warn --download-result=full
python3 - <<'PY'
import hashlib
import json
from pathlib import Path
manifest = json.loads(Path('evidence/strata-iq3s-manifest.json').read_text())
verified = []
for item in manifest['files']:
    path = Path('models') / Path(item['rfilename']).name
    if path.stat().st_size != item['size']:
        raise RuntimeError(f'Wrong size: {path.name}')
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != item['lfs']['sha256']:
        raise RuntimeError(f'Wrong SHA256: {path.name}')
    verified.append({'file': path.name, 'bytes': path.stat().st_size,
                     'sha256': digest.hexdigest()})
Path('evidence/model-verified.json').write_text(json.dumps({
    'repo': manifest['repo'], 'revision': manifest['revision'],
    'files': verified}, indent=2) + '\n')
PY
printf 'verified\n' > evidence/download.status
trap - EXIT
