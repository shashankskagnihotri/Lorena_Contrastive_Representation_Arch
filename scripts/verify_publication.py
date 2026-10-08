#!/usr/bin/env python3
"""Verify the public snapshot and staged upload, without touching live runs.

Run in a CPU allocation on the original cluster. The checks are also usable on
an ordinary workstation. Only committed/staged public paths are inspected; raw
experiment trees are never traversed. Findings fail visibly with exact paths.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import shutil
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'provenance/publication_verification.json')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'provenance/publication_snapshot.json').read_text())
    adaptation_path = ROOT / 'provenance/portability_adaptations.json'
    adaptations = json.loads(adaptation_path.read_text())
    # The adaptation manifest is kept simple and explicit; do not guess hashes.
    changed = {r['path']: r for r in adaptations['adaptations']}
    checked = 0
    for record in manifest['copied']:
        rel = record['target']
        expected = record['sha256']
        if rel in changed:
            require(changed[rel]['original_sha256'] == expected, f'Adaptation origin mismatch: {rel}')
            expected = changed[rel]['public_sha256']
        require((ROOT / rel).is_file(), f'Missing exported evidence: {rel}')
        require(sha(ROOT / rel) == expected, f'Changed exported evidence: {rel}')
        checked += 1
    supplemental = []
    for record in json.loads((ROOT / 'provenance/completion_receipts.json').read_text())['copied']:
        require(sha(ROOT / record['target']) == record['sha256'], f"Changed completion receipt: {record['target']}")
        supplemental.append(record['target'])
    for record in json.loads((ROOT / 'results/diagnostics/source_sha256_manifest.json').read_text())['records']:
        target = ROOT / 'results/diagnostics' / record['destination']
        require(sha(target) == record['sha256'], f'Changed selected diagnostic: {target}')
        supplemental.append(str(target.relative_to(ROOT)))
    files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    files = [x for x in files if x]
    require(bool(files), 'Stage public files before verifying the upload')
    public_files = set(files)
    forbidden_roots = ('sparse_contrast_benchmark/', 'sparse_contrast_benchmark_large_batch/', 'training_dense_testing_sparse/')
    forbidden_suffixes = {'.pt', '.pth', '.ckpt', '.safetensors', '.pem', '.key'}
    sensitive = [
        re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
        re.compile(rb'\bgh[pousr]_[A-Za-z0-9]{30,}\b'),
        re.compile(rb'\bgithub_pat_[A-Za-z0-9_]{50,}\b'),
        re.compile(rb'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{40,}\b'),
        re.compile(rb'https?://[^\s/@:]+:[^\s/@]+@'),
    ]
    text_extensions = {'.py', '.sh', '.md', '.json', '.jsonl', '.txt', '.yml', '.yaml', '.toml', '.ini', '.cff', '.csv'}
    size_total = 0
    largest = []
    for rel in files:
        path = ROOT / rel
        require(not rel.startswith(forbidden_roots), f'Live runtime staged: {rel}')
        require(path.suffix not in forbidden_suffixes and not path.name.startswith('.env'), f'Forbidden artifact staged: {rel}')
        size = path.stat().st_size
        require(size < 100 * 1024**2, f'File exceeds GitHub normal Git limit: {rel}')
        size_total += size
        largest.append((size, rel))
        if path.suffix in text_extensions:
            raw = path.read_bytes()
            # This verifier necessarily contains its own regex, not a credential.
            if rel != 'scripts/verify_publication.py':
                require(not any(pattern.search(raw) for pattern in sensitive), f'Potential credential requires review: {rel}')
        if rel.startswith('src/') and path.suffix == '.py':
            ast.parse(path.read_text(), filename=rel)
    required_pdfs = {'MNIST_error_by_corruption.pdf', 'CIFAR10_error_by_corruption.pdf', 'latency_vs_sparsity.pdf'}
    require({p.name for p in (ROOT / 'plots').glob('*.pdf')} == required_pdfs, 'Expected exactly three top-level PDFs')
    pdfs = {}
    for name in sorted(required_pdfs):
        path = ROOT / 'plots' / name
        require(path.read_bytes().startswith(b'%PDF-'), f'Invalid PDF header: {name}')
        pdfs[name] = sha(path)
    # Verify navigation in current docs; copied archival READMEs preserve old paths.
    link_issues = []
    current_docs = [ROOT / 'README.md', ROOT / 'NOTICE.md', ROOT / 'provenance/README.md', ROOT / 'experiments/README.md', ROOT / 'plots/README.md', ROOT / 'plots/requested_comparisons/README.md', ROOT / 'analysis/README.md', ROOT / 'results/diagnostics/README.md'] + list((ROOT / 'docs').glob('*.md'))
    link_pattern = re.compile(r'(?<!!)\[[^\]]*\]\(([^)]+)\)')
    for doc in current_docs:
        for link in link_pattern.findall(doc.read_text()):
            if '://' in link or link.startswith('#') or link.startswith('mailto:'):
                continue
            target = link.split('#', 1)[0]
            if target:
                resolved = (doc.parent / target).resolve()
                if not resolved.exists():
                    link_issues.append(f'{doc.relative_to(ROOT)} -> {link}')
                elif resolved.is_file():
                    relative = str(resolved.relative_to(ROOT))
                    if relative not in public_files:
                        link_issues.append(f'{doc.relative_to(ROOT)} -> {link} (not staged for publication)')
    require(not link_issues, 'Broken current documentation links: ' + '; '.join(link_issues))
    tomllib.loads((ROOT / 'pyproject.toml').read_text())
    # Import from a relocated source tree so the original Ceph path cannot satisfy
    # an accidentally retained hard-coded import guard.
    with tempfile.TemporaryDirectory(prefix='lorena-public-import-') as temp:
        relocated = Path(temp) / 'checkout'
        shutil.copytree(ROOT / 'src', relocated / 'src', ignore=shutil.ignore_patterns('__pycache__'))
        code = """import sys,json
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import sparse_contrast.frontend, sparse_contrast.models, sparse_contrast.pipeline
import dense_sparse.adapter
from sparse_contrast.common import ROOT
from dense_sparse.common import PYTHON
assert str(ROOT).startswith(sys.argv[2]), str(ROOT)
assert PYTHON == sys.executable
assert str(Path(sparse_contrast.models.__file__).resolve()).startswith(sys.argv[1])
print(json.dumps({'source':sparse_contrast.models.__file__, 'root':str(ROOT), 'python':PYTHON}))
"""
        import_result = subprocess.check_output([sys.executable, '-c', code, str(relocated / 'src'), str(relocated)], cwd=temp, text=True)
    report = dict(verified_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  verifier_source_sha256=sha(Path(__file__)), initial_export_files_verified=checked, supplemental_files_verified=len(supplemental), staged_files=len(files), staged_logical_bytes=size_total,
                  largest_files=[{'bytes':n,'path':p} for n,p in sorted(largest, reverse=True)[:10]],
                  main_pdf_sha256=pdfs, current_document_links_valid=True,
                  relocated_package_import=json.loads(import_result), credential_pattern_scan='passed; scoped patterns, not an exhaustive guarantee',
                  limits='Does not claim raw artifact availability, completed follow-up timing, or portable archival scheduler replay.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
