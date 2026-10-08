"""Public package paths; immutable execution sources remain in provenance/sources."""
from pathlib import Path
import os

PROJECT = Path(os.environ.get('SPARSE_CONTRAST_PROJECT_ROOT', Path(__file__).resolve().parents[2])).resolve()
BASE_ROOT = Path(os.environ.get('SPARSE_CONTRAST_ROOT', PROJECT / 'experiments' / 'training_sparse_testing_sparse')).resolve()
BASE_SOURCE_HASH = '7ea1b6e8bd5a78b0ad411d0545ce060a5ae3597a2f29d788017f39cbeecf50d0'
BASE_SOURCE = PROJECT / 'provenance' / 'sources' / BASE_SOURCE_HASH
