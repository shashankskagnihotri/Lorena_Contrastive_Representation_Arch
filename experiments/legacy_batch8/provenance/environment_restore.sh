#!/usr/bin/env bash
# Reproduce the recorded Linux x86-64 environment without solving dependencies.
set -euo pipefail
cd "$(dirname "$0")/.."
SCB_ENV_NAME="${1:-sparse-contrast-bench}"
source /ceph/sagnihot/miniconda3/etc/profile.d/conda.sh
python - "$SCB_ENV_NAME" <<'PY'
import json, pathlib, subprocess, sys
name = sys.argv[1]
if '/' in name or name in ('base', 'root', '.', '..'):
    raise SystemExit('Choose a new named conda environment')
existing = json.loads(subprocess.check_output(['conda', 'env', 'list', '--json']))['envs']
if any(pathlib.Path(path).name == name for path in existing):
    raise SystemExit('Refusing to overwrite existing environment: ' + name)
PY
conda create -y -n "$SCB_ENV_NAME" --file conda-explicit.txt
conda run -n "$SCB_ENV_NAME" python -m pip install --no-deps --requirement requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu126
conda run -n "$SCB_ENV_NAME" python -m pip check
