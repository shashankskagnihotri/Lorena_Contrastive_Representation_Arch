# Sparse contrast classification benchmark

A new, self-contained study in `/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark`. The authoritative requested protocol is preserved in [provenance/USER_TASK.txt](provenance/USER_TASK.txt). See [PLAN.md](PLAN.md), [PREPROCESSING.md](PREPROCESSING.md), [ARCHITECTURES.md](ARCHITECTURES.md), [DATA.md](DATA.md), and the evidence-generated [REPORT.md](REPORT.md). `STATUS.json` and `outputs/campaign.json` record execution state. A submitted or running job is not a completed result.

The user revised the scope to MNIST raw plus grayscale contrast, while CIFAR-10 retains raw plus all three contrast representations. `experiment_scope.json` records this active scope. The primary registry has 156 runs: two native-resolution architectures and three seeds, with seven trained configurations for MNIST and 19 for CIFAR-10. The excluded MNIST single-color and opponent runs and their artifacts were removed at the user's request; they contribute no results to this study. It uses a fixed held-out-validation split: MNIST 55,000/5,000 and CIFAR-10 45,000/5,000. Primary checkpoints are the final epoch of the frozen schedule. Pilots and explicitly requested software diagnostics are excluded from primary aggregates. No pretrained weights are used.

## Environment and implementation

The dedicated conda environment is `sparse-contrast-bench`, Python 3.11.13, PyTorch 2.9.1+cu126, torchvision 0.24.1+cu126, timm 1.0.22, TensorBoard 2.20.0. Full transitive pins/builds are in `requirements.lock.txt`, `conda-explicit.txt`, and `environment.yml`; installation commands and output are in `provenance/`. Do not mutate this environment under active workers. From this directory:

```bash
source /ceph/sagnihot/miniconda3/etc/profile.d/conda.sh
conda activate sparse-contrast-bench
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg
export SPARSE_CONTRAST_ROOT="$PWD"
```

On a fresh environment installation, the commands actually used are preserved in `provenance/environment_install.sh`. The checked-in lock files capture the final resolved versions, including TensorBoard dependencies. The first attempted conda command requested an unavailable pip build; the successful installation allowed conda to select pip and then recorded its exact build. Existing environments were preserved.

To recreate the final environment under a new name without dependency solving, run `bash provenance/environment_restore.sh sparse-contrast-bench-reproduction`. This uses exact conda builds, every pinned pip package, and the CUDA 12.6 wheel index. TensorBoard 2.20 requires `pkg_resources`, so setuptools is pinned to 80.9.0; its removal is documented in the [setuptools release notes](https://setuptools.pypa.io/en/stable/history.html#v82-0-0). The dependency was repaired before any production pilot started.

All comparison groups use float32, deterministic algorithms, no TF32, and no compilation. This common policy supports both the available RTX 2080 Ti and newer GPUs without assigning precision according to representation. Each attempt records actual hardware. The launcher includes the eligible H100 partition and other authorized accelerator partitions; Slurm selects placement. All eight H100s were allocated when this campaign was prepared, so no productive H100 migration was available. It never uses SSH-visible GPUs without an allocation.

The fixed frontend runs on raw [0,1] values, with source-native signed outputs. Each representation has one population mean/std vector fitted on its full clean training partition at 0% thresholding. The fitted artifacts in `normalization/` are frozen across seeds, architectures, and sparsities. The primary token epsilon remains zero. Native coefficient ranking occurs before normalization. Compact paths preserve the original support and coordinates. The equivalent dense-normalize-then-gather implementation is used at the input boundary; thresholded coefficients inside retained patches retain their affine zero offsets. See the source-parity documentation for the historical application's input-normalization and grayscale-adapter differences.

## Data and required correctness checks

Existing originals under `/ceph/sagnihot/datasets/` are reused. The initial preparation downloaded only the missing official MNIST-C archive, verified its publisher checksum, and extracted safely. Worker loaders use `datasets_manifest.json`, never rediscover the dataset tree.

```bash
python prepare_data.py --data-root /ceph/sagnihot/datasets/ --download
python compute_normalization_stats.py --data-root /ceph/sagnihot/datasets/ --device cpu --threads 4 --output-dir normalization
SPARSE_BENCH_TEST_DEVICE=cuda python -m pytest tests
python scripts/check_training.py --device cuda --dataset all --architecture all
python scripts/preprocessing_pilot.py --device cuda
python smoke.py --device cuda
```

Run compute-intensive commands inside a Slurm allocation. The scripts `scripts/verify_core.sh`, `scripts/overfit.sh`, `scripts/verify_runtime.sh`, and `scripts/verify_end_to_end.sh` encode the actual verification sequence. These diagnostics were explicitly requested; their tiny subsets do not substitute for full pilots or primary experiments. The full-training preprocessing diagnostic reports patch sizes 1, 2, and 4 without adding a training sweep.

Tests cover source parity; exact coefficient ties and support; full-split statistics and cache identities; dense architecture and compact/dense-masked forward/gradient parity; shifted boundaries, odd merging, empty images, original positions; deterministic updates and serialized epoch-boundary continuation; metrics, event consistency, timing boundaries, and controller behavior. GPU tests require a real allocated device. Core and integration JUnit files and diagnostic completion artifacts live in `outputs/checks/`.

## Full execution, status, and recovery

```bash
./run_all.sh /ceph/sagnihot/datasets/
python -m sparse_contrast.campaign status
```

`run_all.sh` respects data, source, statistics, correctness, overfit, and smoke dependencies. After those gates, the durable Slurm controller starts four full raw-input pilots: MNIST 50 epochs and CIFAR-10 200 epochs, one per architecture, seed 0. A prespecified clean-validation gate requires complete finite runs, decreasing training loss, and final validation accuracy at least 97% for MNIST or 65% for CIFAR-10. A failure blocks recipe freeze for diagnosis, with no test/corruption feedback. These are convergence gates, not claims of optimal performance. The four pilots are additional to the 156 fresh main/control runs.

As each dataset/architecture pilot passes, its recipe is frozen and its fresh main runs (21 for MNIST, 57 for CIFAR-10) are submitted in a deterministic shuffled order while other groups continue. A failed group remains blocked without stopping valid groups. All 156 evaluations and four passed pilot gates are required before latency measurement. Each run uses batch/effective batch 8, no accumulation, AdamW3e-4, cosine schedule, specified warmup, clipping1, and stochastic depth0.1. CIFAR crop/flip decisions and sample order are keyed independently of model RNG. Initialization is keyed by parameter name and seed so shared backbone shapes match across different native input-channel counts.

Production source files are copied to a content-addressed immutable snapshot before workers start. Verification jobs record their attempt identity; the repaired smoke path also executes a frozen snapshot. Configurations include source, environment, data-manifest and normalization identities. An epoch checkpoint contains model, optimizer, scheduler description, step/epoch/sampler boundary, and Python/NumPy/PyTorch CPU/CUDA RNG state. Resume replays an unfinished epoch from its valid boundary. Orphaned numeric logs are preserved separately; step and epoch event streams have separate rollback axes. There is no claim of bitwise reproducibility across GPU types.

```bash
python train.py --config /absolute/path/to/config.json
python train.py --config /absolute/path/to/config.json --resume /absolute/path/to/latest.pt
python -m sparse_contrast.campaign stop
python -m sparse_contrast.campaign resume
```

The normal training command resumes the same run's `latest.pt`. Explicit missing/incompatible checkpoints fail. Stop creates the project `STOP` marker and cancels only identified campaign workers/controllers; resume behavior and lock reclamation require terminal Slurm ownership evidence. No unrelated job is cancelled. Read the controller's status before acting; never manually delete an active writer lock. See `outputs/campaign.json`, `outputs/slurm/`, and persistent submission journals for exact identities. Deterministic code failures stop affected work; bounded transient infrastructure recovery does not silently turn a failure into success.

The controller can submit, reconcile, retry narrowly classified infrastructure failures, continue wall-time-limited work, and chain evaluation/timing/report phases. It does not perform arbitrary intelligent code repair. Its own scheduler-supported handoff preserves supervision beyond one controller allocation. Review the actual heartbeat/job state before claiming it remains active.

`outputs/runtime_estimate.json` is refreshed by the controller from completed full-pilot epochs, including full validation and recorded epoch logging. It projects training device-hours for 156 main runs plus four pilots using each group's observed GPU class, with idealized four/eight-GPU elapsed scenarios. It explicitly excludes final evaluation, latency measurement, queues, setup/checkpoint costs, and unknown contrast execution cost differences. It is a provisional training reference, not a campaign completion forecast; missing pilot evidence stays missing.

## Evaluation, latency, and reporting

```bash
python evaluate.py --config /absolute/path/to/config.json
python benchmark.py --config /absolute/path/to/config.json --execution compact --batch-size 1 --device cuda
python -m sparse_contrast.campaign launch
python report.py
python check_tensorboard.py --all
```

Final evaluation uses every official 10,000-image clean test set and all released primary corruption cells: MNIST-C 15 fixed-severity cells, CIFAR-10-C 15 corruptions x5 severities. Cell JSON stores integer counts, full per-sample identities/predictions/correctness NPZ files, and token records. Cell completion is independently verified before aggregation. No official test result selects a checkpoint or recipe.

The latency controller first freezes a GPU model/backend contract, then schedules 24 independent dataset/architecture/seed/batch blocks. MNIST blocks each contain 12 matched execution workers and CIFAR-10 blocks each contain 34, for 552 workers total. Each block runs its workers sequentially in fresh processes on one reserved GPU; independent blocks can run concurrently on GPUs of the same model. A continuation may use a different physical GPU of that class, with every cell's UUID and session retained. The fixed real batches and randomized method order are saved. Every trained condition and every compact checkpoint's dense-masked path is measured at batches1 and8. Each clean/corruption cell has 50 warmups and200 measured paired batches, plus a separate sustained-throughput pass. The online GPU and pinned-host scopes include frontend, ranking, support, normalization and compact model work. Loader-to-CPU and cached-model-only diagnostics have separate names/boundaries. No TensorBoard writer, image rendering or profiler is active during headline timings. Profiler passes record CPU/CUDA scopes and shapes separately; inclusive stage intervals are not summed into an invented end-to-end total.

The reporting code validates the exact active registry and excludes withdrawn MNIST representations from normalization, preprocessing, curves, and comparisons. It reads durable evidence, writes `outputs/tables/`, and renders matplotlib/seaborn PDF/SVG/300-dpi PNG figures and plot-data CSV files. It preserves absent cells as missing and keeps seed variability separate from repeated-batch timing variation. `REPORT.md` opens with clean/corruption and online-latency comparisons; incomplete results are explicitly marked. MNIST background sparsity prevents uniquely attributing speedups to contrast without a separate raw-empty-patch control.

## TensorBoard

Within an authorized allocation, from the project directory:

```bash
tensorboard --logdir outputs/tensorboard --host 127.0.0.1 --port 6006
```

Use an SSH tunnel to that actual allocated host, for example `ssh -N -L 6006:127.0.0.1:6006 -J sagnihot@dws-login-02.informatik.uni-mannheim.de sagnihot@ALLOCATED_NODE`, then open `http://127.0.0.1:6006`. Do not expose a public unauthenticated dashboard. `scripts/check_tensorboard_server.py` tests localhost startup and stops its own server afterwards. Training, epoch, evaluation, and latency axes have separate event paths. Every run logs exact sample-weighted metrics, normalization identities, support/execution information, bounded histograms, signed diagnostic panels, and full validation each epoch. Final scientific figures are mirrored to `study_summary`.

Optional full-data refits are disabled. After primary recipes freeze, separate full-data normalization artifacts can be fitted with `--protocol full_train_refit`; `python configure_refit.py --enable` writes separate refit configurations. These have 60,000/50,000 training images and no independent validation curve and are excluded from primary held-out aggregates. No full-data refits are launched by default.
