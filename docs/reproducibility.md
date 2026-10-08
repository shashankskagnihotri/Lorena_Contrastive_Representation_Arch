# Reproducibility and artifact boundaries

The public checkout supports rebuilding the published comparison figures from saved numerical evidence and importing the scientific implementation. Repeating model training, reproducing inference from the exact checkpoints, and resuming the original cluster controllers additionally require external artifacts and the original execution context. These are different levels of reproduction; importing a package is not evidence that a registered campaign can run elsewhere.

## Rebuild the three comparison PDFs

From the repository root, use a separate analysis environment with the recorded plotting versions:

```bash
python -m venv .venv-analysis
source .venv-analysis/bin/activate
python -m pip install numpy==2.2.6 matplotlib==3.10.7 seaborn==0.13.2
python analysis/reproduce_figures.py --input-dir plots --output-dir reproduced_plots
```

The two directory arguments have the defaults shown. This command reads the committed aggregate and seed CSVs and writes exactly these comparison PDFs into the output directory:

- `MNIST_error_by_corruption.pdf`: both architectures, clean error and all 15 MNIST-C corruptions.
- `CIFAR10_error_by_corruption.pdf`: both architectures, clean error and all 15 CIFAR-10-C corruptions, with equal averaging over five severities within each seed.
- `latency_vs_sparsity.pdf`: both datasets and architectures, with batch 1 and batch 8 on separate pages.

The [renderer](../analysis/reproduce_figures.py) verifies expected condition and seed coverage and recomputes saved means and sample standard deviations before plotting. It reads paths supplied as input directories, not the historical absolute paths embedded in evidence rows. It requires neither Torch nor a GPU, images, or trained checkpoints. Run rendering in an authorized CPU allocation when working on a cluster. `pypdfium2==4.30.0` is an optional dependency for inspecting rendered PDF pages, not a requirement for producing them.

The plotted numbers are a saved observation snapshot. The follow-up latency curves deliberately retain missing points where three seeds were not yet available at the declared cutoff. Rerendering cannot turn incomplete measurements into complete ones and does not query live jobs. PDF file hashes can change with metadata, font, or backend differences; numerical agreement and the recorded rendering environment are the relevant checks when regenerating the figures. The original PDFs, CSVs, and provenance remain under [plots](../plots/), with their definitions in [the comparison README](../plots/requested_comparisons/README.md).

The programs under [analysis/original_exporters](../analysis/original_exporters/) preserve the original exporters used on the research filesystem. They retain cluster paths and can depend on full local tables omitted from Git. Use `analysis/reproduce_figures.py` for public figure reconstruction.

## Install the scientific package in a matching environment

The recorded scientific stack uses Python 3.11.13, PyTorch 2.9.1 with its CUDA 12.6 build, torchvision 0.24.1 with CUDA 12.6, timm 1.0.22, and NumPy 2.2.6. The full environment records are:

- [requirements.lock.txt](../provenance/environment/requirements.lock.txt): installed Python package versions, including local CUDA build suffixes.
- [environment.yml](../provenance/environment/environment.yml): exported Conda environment and its pip dependencies.
- [conda-explicit.txt](../provenance/environment/conda-explicit.txt): platform-specific Conda package locations.

These records describe the actual environment. Restoring them on a different operating system or GPU stack may require platform-specific installation work; the Conda explicit file is not a cross-platform lock. A CUDA-enabled wheel also does not supply a compatible host GPU driver. Keep an existing production environment unchanged and restore a separate environment when needed.

Once the matching environment is available, install only this checkout:

```bash
python -m pip install --no-deps --no-build-isolation -e .
```

`--no-deps` prevents the package installation from replacing the restored scientific dependencies. `--no-build-isolation` uses the environment's installed setuptools and wheel instead of preparing another build environment. The exact environment export includes both. The pinned requirements in [pyproject.toml](../pyproject.toml) describe direct dependencies; the complete lock also fixes transitive packages and the original CUDA wheel variants.

The public packages are named `sparse_contrast` and `dense_sparse`. The modules containing the frontend, normalization, compact models, and inference adapter can be imported for inspection and new explicitly configured work. Installation does not download datasets or trained weights, restore an output registry, or validate an old checkpoint automatically.

## Public package paths versus immutable execution sources

The original training, measurement, inference, numerical-validation repair, and controller-repair source trees are retained under [provenance/sources](../provenance/sources/), keyed by their original source hashes. They are the authoritative record of what executed. [publication_snapshot.json](../provenance/publication_snapshot.json) binds the initial exported files to their original paths and SHA-256 values.

The importable `src/` packages were copied from those snapshots. Exactly three files then received public bootstrap adaptations:

| Public file | Adaptation |
|---|---|
| `src/dense_sparse/__init__.py` | Resolve the checkout and study directories without a hardcoded Ceph guard, environment mutation, or insertion of an archived snapshot into `sys.path`. Keep the original `BASE_SOURCE_HASH`; `BASE_SOURCE` is an archival reference. |
| `src/dense_sparse/common.py` | Default to the public follow-up experiment directory and use `sys.executable` for the current Python interpreter. |
| `src/sparse_contrast/common.py` | Default to the public main experiment directory. |

[portability_adaptations.json](../provenance/portability_adaptations.json) records the initial export hash, original archived hash, new public hash, and reason for each change. It also binds the publication snapshot hash. Every function and class in these three files is unchanged at the Python AST level; only module bootstrap, paths, imports, and interpreter selection differ. No frontend weights, pruning rules, normalization, model operations, training logic, timing loop, or provenance-validation function was changed. The full archived originals remain byte-preserved.

An editable installation derives the project root from its checkout. Explicit overrides are available:

| Environment variable | Default |
|---|---|
| `SPARSE_CONTRAST_PROJECT_ROOT` | Checkout containing `src/` |
| `SPARSE_CONTRAST_ROOT` | `<project>/experiments/training_sparse_testing_sparse` |
| `DENSE_SPARSE_ROOT` | `<project>/experiments/training_dense_testing_sparse` |

For a noneditable installation, set `SPARSE_CONTRAST_PROJECT_ROOT` to the checkout containing the experiment and provenance directories before importing the packages. A wheel contains package code; it does not relocate or bundle all experiment artifacts. Explicit study-root overrides take precedence over the project default. These settings locate artifacts; they do not rewrite source identities or make old absolute checkpoint paths valid.

The public packages are intentionally **not** registered as a replacement execution source for completed production cases. For example, `dense_sparse.common.verify_source()` still requires an immutable inference source manifest and verifies every registered byte. Calling production validation from the adapted public tree therefore fails honestly when that immutable source context is absent or different. Do not bypass this by disabling checks, inventing a matching manifest, dropping state-dictionary keys, or loading a checkpoint with `strict=False`.

## What is included and what remains external

The checkout includes source snapshots, resolved configurations and manifests, frozen normalization artifacts, environment records, aggregate and seed tables, selected diagnostic evidence, and the published figures. [repository_layout.md](repository_layout.md) explains which directory is authoritative for each kind of artifact. [local_artifact_inventory.json](../provenance/local_artifact_inventory.json) describes the external local trees at the inventory time; it is not an atomic full-content hash of a still-running campaign.

The ordinary Git repository does not contain the original image datasets, final/latest model checkpoint blobs, exhaustive per-image prediction arrays, all raw timing and profiler artifacts, TensorBoard event streams, or complete scheduler logs. Some large tables were also omitted explicitly in `publication_snapshot.json`. A completion receipt or checkpoint hash identifies a particular trained model but cannot reconstruct its weights. Public summary tables can reproduce their published aggregations and figures; they cannot independently establish predictions for an unavailable checkpoint.

Reproducing the original learned-model results requires access to the exact checkpoint files and their completion receipts, registered configuration and source snapshots, matching frozen normalization artifacts, and the official dataset versions and split identities. Dataset preparation and coverage are documented in the archived [DATA.md](../experiments/training_sparse_testing_sparse/DATA.md), [dataset manifest](../experiments/training_sparse_testing_sparse/datasets_manifest.json), and current [experiment protocol](experiment_protocol.md). Historical absolute paths in these records are preserved as evidence, not silently remapped.

Independent retraining additionally requires the complete resolved training recipes, seeds, initialization, keyed augmentation and sample-order rules, all epochs, and the prescribed validation split. A new execution should have a new source/configuration identity and output location. It must not overwrite the completed study or claim exact-resume continuity from a weights-only load. The reported protocol is `heldout_val`, with 55,000 MNIST or 45,000 CIFAR-10 training images; the official test and corruption sets do not select recipes.

## Archived controllers are not portable rerun commands

The copied controller and scheduler code describes the execution on the original cluster. It depends on Slurm, site partition/account policies, original absolute artifact paths, submission receipts, locks, and the registered campaign state. It is retained to explain the durable execution and failure-recovery protocol. It is not a general launcher that can safely be pointed at the public experiment directories and expected to resume the original jobs.

On the original cluster, the existing controller and frozen source versions remain authoritative. The public export does not move those directories, replace the active interpreter, mutate live state, or start a second writer. Reusing the architecture on another system requires a deliberately registered new experiment and a scheduler adapter appropriate to that system. That work is separate from this publication export.

Timing reproduction has a stricter boundary than accuracy evaluation. The primary comparisons use an isolated NVIDIA RTX A6000, AMD EPYC 7413 CPU class, the recorded thread/backend settings, FP32, the exact sample panels and checkpoints, 50 warmups, and 200 timed calls per scope. New hardware or another CPU/GPU/thread class defines a separate timing result, even if the model code is identical. Do not pool the older hardware cohort or substitute a newly measured batch size into the published comparison. The detailed boundaries and memory policy are in [method.md](method.md).

## Known unexecuted full-refit path

The optional full-training-set refit path has a protocol-name mismatch in the archived implementation. Dataset/statistics preparation uses canonical `full_refit`; the trainer suppresses validation for `full_train_refit`, while normalization lookup uses the configuration protocol literally. Canonical `full_refit` can therefore try to load a forbidden validation split, and the alternate spelling can seek a normalization artifact under the wrong name. This path was not used by the reported `heldout_val` campaign.

The presence of `configure_refit.py` is not a claim that full refitting was successfully run. A future full refit needs an explicit protocol/path repair, a new recorded source identity, and validation of its complete data/normalization/training flow. The publication adaptations do not fix or conceal this dormant scientific-workflow issue.
