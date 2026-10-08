#!/usr/bin/env bash
set -euo pipefail
source /ceph/sagnihot/miniconda3/etc/profile.d/conda.sh
conda create -y -n sparse-contrast-bench python=3.11.13 pip
conda activate sparse-contrast-bench
conda install -y --freeze-installed setuptools=80.9.0
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.in
python -m pip freeze --all > provenance/pip-freeze.txt
python - <<'PY'
from importlib.metadata import version
from pathlib import Path
lines = Path('provenance/pip-freeze.txt').read_text().splitlines()
# Conda's pip metadata names an unavailable build-machine path; its exact build
# remains in conda-explicit.txt and its installed version is pinned here.
Path('requirements.lock.txt').write_text('\n'.join(
    'pip==' + version('pip') if line.startswith('pip @ file:') else line
    for line in lines) + '\n')
PY
conda list --explicit > conda-explicit.txt
conda env export > environment.yml
