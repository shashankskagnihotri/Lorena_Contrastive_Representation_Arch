#!/usr/bin/env bash
set -euo pipefail
python scripts/check_training.py --device cuda --dataset all --architecture all
