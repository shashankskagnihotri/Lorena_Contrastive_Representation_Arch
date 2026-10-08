"""Evaluation-only follow-up, importing the immutable original experiment implementation."""
from pathlib import Path
import os
import sys

PROJECT = Path('/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch')
BASE_ROOT = PROJECT / 'sparse_contrast_benchmark_large_batch'
BASE_SOURCE_HASH = '7ea1b6e8bd5a78b0ad411d0545ce060a5ae3597a2f29d788017f39cbeecf50d0'
BASE_SOURCE = BASE_ROOT / 'outputs/source' / BASE_SOURCE_HASH
if os.environ.get('SPARSE_CONTRAST_ROOT', str(BASE_ROOT)) != str(BASE_ROOT):
    raise RuntimeError('Original checkpoint/data root must remain unchanged')
os.environ['SPARSE_CONTRAST_ROOT'] = str(BASE_ROOT)
if not BASE_SOURCE.is_dir():
    raise FileNotFoundError(BASE_SOURCE)
sys.path.insert(0, str(BASE_SOURCE))
