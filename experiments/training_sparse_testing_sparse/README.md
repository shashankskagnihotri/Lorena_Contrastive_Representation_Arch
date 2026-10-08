# Sparse contrast classification benchmark: larger-batch rerun

A new, self-contained study in `/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch`. The original requested protocol is preserved in [provenance/USER_TASK.txt](provenance/USER_TASK.txt). The October 2 larger-batch authorization and campaign separation are recorded in [CAMPAIGN_AMENDMENT.md](CAMPAIGN_AMENDMENT.md), which overrides the original training batch size. See [PLAN.md](PLAN.md), [PREPROCESSING.md](PREPROCESSING.md), [ARCHITECTURES.md](ARCHITECTURES.md), [DATA.md](DATA.md), and the evidence-generated [REPORT.md](REPORT.md). `STATUS.json` and `outputs/campaign.json` record execution state. A submitted or running job is not a completed result.

The user revised the scope to MNIST raw plus grayscale contrast, while CIFAR-10 retains raw plus all three contrast representations. `experiment_scope.json` records this active scope. The primary registry has 156 runs: two native-resolution architectures and three seeds, with seven trained configurations for MNIST and 19 for CIFAR-10. The excluded MNIST single-color and opponent runs and their artifacts were removed at the user's request; they contribute no results to this study. It uses a fixed held-out-validation split: MNIST 55,000/5,000 and CIFAR-10 45,000/5,000. Primary checkpoints are the final epoch of the frozen schedule. Pilots and explicitly requested software diagnostics are excluded from primary aggregates. No pretrained weights are used.

The existing batch-8 campaign and its checkpoints/results remain under the separate original `sparse_contrast_benchmark` directory. This rerun starts fresh training and independent pilots; no old trained run, checkpoint, controller, training gate receipt, or accuracy/timing aggregate is copied here. Invariant preprocessing diagnostics were later reused only after the identity checks documented below. Raw datasets are shared, while split/statistics metadata and their checksummed files are copied locally. All four preferred resource profiles and group selections verified before launch. By the October 4 audit, all four pilots and all 156 main training/evaluation runs had completed. Isolated latency and final reporting remain unfinished. The original campaign retains its `STOP` marker and archived evidence, and its jobs have drained from the scheduler queue.

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

All comparison groups use float32 and call `torch.use_deterministic_algorithms(True)`. cuDNN benchmarking is disabled, cuDNN deterministic mode is enabled, TF32 is disabled, and compilation is not used. This common policy supports both the available RTX 2080 Ti and newer GPUs without assigning precision according to representation. Each attempt records actual hardware. Training placement follows each selected group's recorded memory requirement and uses Slurm allocations. The October 4 resource audit found that this user's authorized `ml-staff` account cannot use the H100 partition restricted to `dws-staff`; a partition appearing in launcher preferences does not establish access. The active timing deployment therefore uses authorized RTX A6000 GPUs. It never uses SSH-visible GPUs without an allocation.

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

Run compute-intensive commands inside a Slurm allocation. The scripts `scripts/verify_core.sh`, `scripts/overfit.sh`, `scripts/verify_runtime.sh`, and `scripts/verify_end_to_end.sh` encode the actual verification sequence. These diagnostics were explicitly requested; their tiny subsets do not substitute for full pilots or primary experiments. The full-training preprocessing diagnostic reports patch sizes 1, 2, and 4 without adding a training sweep. Its saved evidence is now present in `outputs/preprocessing_pilot/`: 60 active-scope rows and per-image arrays covering all 55,000 MNIST and 45,000 CIFAR-10 training samples. [Reuse provenance](provenance/reused_preprocessing_2026-10-04.json) records matching scientific code, raw-data identities, exact splits, normalization fits, and recomputed saved-array statistics. Existing shared previews were independently checked in place: four active fits, 40 NPZ tensor/mask files, and 160 PNG/PDF pairs with fixed IDs and training-derived display scales. No withdrawn MNIST color artifact was copied. Eight supplemental preprocessing figures were rendered from these inputs without retraining or rerunning the full report.

Supplement job 356618 completed with exit 0:0 in 24 seconds after successful report job 356058. It used frozen reporter source `7ea1b6e8bd5a78b0ad411d0545ce060a5ae3597a2f29d788017f39cbeecf50d0` and supplement script SHA256 `dc1cd8e2477168cf7efe3276bb8aa19e76380a4aa54cf3eb7975ce4a815d3ce1`. All eight active-scope preprocessing figures are now linked in the report and coverage index, giving 376 unique PNG/PDF/SVG/CSV figure bundles. Every bundle has a parseable nonempty plot-data CSV, and all 376 expected image tags were verified in the six TensorBoard study-summary groups, including all eight preprocessing images. Representative MNIST grayscale and CIFAR color-opponency exports were visually checked for readable axes, legends, and complete patch/sparsity settings. The separately frozen postprocessor added explicit hardware IDs and GPU/CPU cohort labels to the partial timing headlines. Original report bytes and SHA receipts remain under `outputs/report_annotations/`; the [completion receipt](outputs/checks/preprocessing_report_supplement.json) records figure paths, source and script hashes, reused-evidence hash, and the annotation receipt. The [launch receipt](outputs/checks/preprocessing_report_supplement_launch.json) records the exact command inputs and successful-report dependency. Accuracy reporting is complete; the full A6000 timing campaign remains active.

The scheduled command, from this project root in the pinned environment with `SPARSE_CONTRAST_ROOT` set to this root, is:

```bash
PYTHONPATH="$PWD/outputs/source/7ea1b6e8bd5a78b0ad411d0545ce060a5ae3597a2f29d788017f39cbeecf50d0" \
python outputs/report_annotations/source/dc1cd8e2477168cf7efe3276bb8aa19e76380a4aa54cf3eb7975ce4a815d3ce1/supplement_preprocessing_report.py
```

Job 356618 successfully executed this command; it is recorded for reproduction.

Tests cover source parity; exact coefficient ties and support; full-split statistics and cache identities; dense architecture and compact/dense-masked forward/gradient parity; shifted boundaries, odd merging, empty images, original positions; deterministic updates and serialized epoch-boundary continuation; metrics, event consistency, timing boundaries, and controller behavior. GPU tests require a real allocated device. Core and integration JUnit files and diagnostic completion artifacts live in `outputs/checks/`.

## Full execution, status, and recovery

The frozen training/evaluation source is `outputs/source/9359e4f10c0c2184eef4af8885edc7076bded30188775cc14222f79819f648c4`. Verification job 355427 passed 140 tests; final-source job 355440 passed 121 tests. These suites overlap and are not 261 distinct tests. `outputs/checks/completed.json` records both receipts, hashes, the verified four-group selection, and explicitly labeled historical evidence for unchanged mathematical components. The manifest base batch is null; all four `recipe_by_group` overrides are mandatory.

Status snapshot, October 4 at 19:36 CEST: all 156 main runs have verified final training and full clean/corruption evaluation evidence. The four separate raw pilots also completed. Main coverage is:

| Dataset | Architecture | Training completed | Full evaluation completed |
|---|---|---:|---:|
| MNIST | ViT-Small | 21/21 | 21/21 |
| MNIST | Swin-Tiny | 21/21 | 21/21 |
| CIFAR-10 | ViT-Small | 57/57 | 57/57 |
| CIFAR-10 | Swin-Tiny | 57/57 | 57/57 |

The audit checked all 24,900 epoch records, exact training/validation counts and configuration identities, and all 9,336 official evaluation cells: 156 clean plus 9,180 corruption cells, totaling 93,360,000 image predictions. Every saved prediction NPZ hash matched its receipt, cell counts and confusion matrices reconciled, and corruption means were recomputed. No main training/evaluation failure record was found. The read-only audit did not rehash the large final checkpoints or token JSONL files; their recorded identities were checked. The [complete three-seed accuracy CSV](../debugging/completed_main_conditions_large_batch_2026-10-04.csv) contains all 52 retained conditions, exact group recipes, final online train/held-out validation/clean metrics, mCE, sample standard deviations, paired changes, and source paths. Its absolute location is `/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/debugging/completed_main_conditions_large_batch_2026-10-04.csv`. As report tables were produced, all 52 clean-accuracy/mCE means and sample standard deviations matched this independent CSV within 1e-12; representative MNIST plots were visually inspected for labels, scales, scope, and clipping.

Accuracy completion does not complete the campaign. Nine workers and 684 cells from the original RTX 5000 Ada timing cohort are preserved as partial evidence. The fresh RTX A6000 cohort launched after the final resource gate passed. At 19:53 CEST, controller 356059 and 31 isolated GPU workers (356061 through 356091) were verified running on `dws-14` through `dws-17`; remaining workers were being submitted. This establishes an active timing campaign, not completed timing coverage. Use `STATUS.json`, `outputs/campaign.json`, and scheduler receipts for current execution state.

```bash
./run_all.sh /ceph/sagnihot/datasets/
python -m sparse_contrast.campaign status
```

`run_all.sh` respects data, source, statistics, correctness, overfit, and smoke dependencies. After those gates, the durable Slurm controller starts four full raw-input pilots: MNIST 50 epochs and CIFAR-10 200 epochs, one per architecture, seed 0. A prespecified clean-validation gate requires complete finite runs, decreasing training loss, and final validation accuracy at least 97% for MNIST or 65% for CIFAR-10. A failure blocks recipe freeze for diagnosis, with no test/corruption feedback. These are convergence gates, not claims of optimal performance. The four pilots are additional to the 156 fresh main/control runs.

As each dataset/architecture pilot passes, its recipe is frozen and its fresh main runs (21 for MNIST, 57 for CIFAR-10) are submitted in a deterministic shuffled order while other groups continue. A failed group remains blocked without stopping valid groups. All 156 evaluations and four passed pilot gates are required before latency measurement. Each verified group has one physical training batch, the same effective batch, and the same validation/test batch across all its conditions and seeds. `experiment_manifest.json` records the accepted values in `recipe_by_group`. AdamW uses base LR `3e-4 * sqrt(B / 128)`, betas `(0.9, 0.999)`, weight decay 0.05, cosine decay to 1e-6, gradient clipping 1, and stochastic depth 0.1. There is no gradient accumulation. Training and evaluation both use zero data-loader workers. MNIST retains 50 epochs with 5 warmup epochs; CIFAR-10 retains 200 with 10. This changes optimizer-update counts and is not an exact equivalent of batch-8 training. CIFAR crop/flip decisions and sample order are keyed independently of model RNG. Initialization is keyed by parameter name and seed so shared backbone shapes match across different native input-channel counts.

The verified profile choices available at this update are:

| Dataset | Architecture | Train / evaluation batch | Base LR | Training GPU placement | Profile state |
|---|---|---:|---:|---|---|
| MNIST | ViT-Small | 256 | 0.000424264 | 48 GB class or larger | Complete |
| MNIST | Swin-Tiny | 1024 | 0.000848528 | 11 GB class or larger | Complete |
| CIFAR-10 | Swin-Tiny | 512 | 0.000600000 | 11 GB class or larger | Complete |
| CIFAR-10 | ViT-Small | 128 | 0.000300000 | 48 GB class or larger | Complete, job 355430 |

The controller validated all four groups' condition coverage, input/source hashes, finite updates, and memory headroom in `outputs/batch_profile/selection.json` before this launch. It can release a verified group's full pilot while another group's profile remains pending; an unselected group cannot inherit the placeholder batch. Pilot validation gates still control release of that group's main runs. Profile measurements characterize training capacity and short-run throughput, not accuracy or isolated inference latency. See [CAMPAIGN_AMENDMENT.md](CAMPAIGN_AMENDMENT.md) for exact profile evidence and selection rules.

Production source files are copied to a content-addressed immutable snapshot before workers start. Verification jobs record their attempt identity; the repaired smoke path also executes a frozen snapshot. Configurations include source, environment, data-manifest and normalization identities. An epoch checkpoint contains model, optimizer, scheduler description, step/epoch/sampler boundary, and Python/NumPy/PyTorch CPU/CUDA RNG state. Resume replays an unfinished epoch from its valid boundary. Orphaned numeric logs are preserved separately; step and epoch event streams have separate rollback axes. There is no claim of bitwise reproducibility across GPU types.

```bash
python train.py --config /absolute/path/to/config.json
python train.py --config /absolute/path/to/config.json --resume /absolute/path/to/latest.pt
python -m sparse_contrast.campaign stop
python -m sparse_contrast.campaign resume
```

The normal training command resumes the same run's `latest.pt`. Explicit missing/incompatible checkpoints fail. Stop creates the project `STOP` marker and cancels only identified campaign workers/controllers; resume behavior and lock reclamation require terminal Slurm ownership evidence. No unrelated job is cancelled. Read the controller's status before acting; never manually delete an active writer lock. See `outputs/campaign.json`, `outputs/slurm/`, and persistent submission journals for exact identities. Deterministic code failures stop affected work; bounded transient infrastructure recovery does not silently turn a failure into success.

The controller can submit, reconcile, retry narrowly classified infrastructure failures, continue wall-time-limited work, and chain evaluation/timing/report phases. It does not perform arbitrary intelligent code repair. Its own scheduler-supported handoff preserves supervision beyond one controller allocation. Review the actual heartbeat/job state before claiming it remains active. Bounded monitor job 356616 is verified RUNNING in `ACTIVE_REPAIR` mode with a fresh heartbeat and no startup incidents or consumed agent calls. It replaces job 356613, which stopped cleanly after a terminal-job lookup gap was found. The corrected receipt fallback passed 19 tests in job 356615 and requires an exact job ID, current user, and job name, rejecting conflicting comments. The restart preserved the deadline of October 8 at 19:54:42 CEST and all call counters. Limits remain 96 hours, five-minute checks, eight total agent calls, three calls per incident, and 45 minutes per call; the Slurm allocation is four CPUs, 8 GiB, and 97 hours. The [latency startup audit](outputs/checks/latency_startup_2026-10-04.json) records 31 GPU workers, 1,014 completed cell receipts at its snapshot, and all 552 registered workers. This is running bounded supervision, not a claim that the remaining benchmark is complete.

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

The October 4 timing amendment launched the separate `a6000-2026-10-04` cohort. Its [hardware contract](outputs/benchmarks/cohorts/a6000-2026-10-04/hardware_contract.json) fixes NVIDIA RTX A6000, AMD EPYC 7413 CPUs, four PyTorch intra-op threads, 48 inter-op threads, float32, and the recorded driver/backend versions. The selected pool contains 31 A6000 GPUs on matching `dws-14` through `dws-17`; the two `dws-10` devices are excluded because that host has a different CPU configuration. The original RTX 5000 Ada cohort remains separate, with nine complete workers and 684 cells. Its values cannot fill missing A6000 comparisons. Different GPU types are never pooled.

Combined orchestration tests passed in job 356054 (179 tests), and final resource/host/monitor checks passed in job 356057 (112 tests; overlapping suites). Timing and orchestration execute frozen source `7ea1b6e8bd5a78b0ad411d0545ce060a5ae3597a2f29d788017f39cbeecf50d0`. Training/evaluation keep their original source identity `9359e4f10c0c2184eef4af8885edc7076bded30188775cc14222f79819f648c4`. Controller 356059 was verified running on `dws-06` with a fresh heartbeat. Preparation job 356060 completed and verified isolation before worker launch.

The 24 dataset/architecture/seed/batch blocks define matched comparisons and coverage. MNIST blocks each contain 12 execution workers and CIFAR-10 blocks each contain 34: 552 workers and 33,312 clean/corruption cells. Under the A6000 amendment, every worker gets its own isolated Slurm GPU allocation and fresh process, so workers within a comparison block can run concurrently. All must match the frozen GPU, CPU, thread, and backend contract. Block completion is reconciled only after all constituent workers verify. Each cell retains its physical-device UUID and measurement session. The fixed real batches and randomized dispatch order are saved. Every trained condition and every compact checkpoint's dense-masked path is measured at batches 1 and 8, with 50 warmups and 200 measured paired batches per cell, plus a separate sustained-throughput pass. The online GPU and pinned-host scopes include frontend, ranking, support, normalization, and compact-model work. Loader-to-CPU and cached-model-only diagnostics have separate boundaries. No TensorBoard writer, rendering, or profiler is active during headline timings. Separate profiler passes record CPU/CUDA scopes and shapes; inclusive stage intervals are not summed into an end-to-end total.

Interim report job 356049 was cancelled after reaching about 7.6 GiB under an 8 GiB allocation. Replacement accuracy-status report job 356058 completed with exit 0:0 in 45 minutes 38 seconds, using a 96 GiB/12-hour allocation and reaching about 41.6 GiB maximum resident memory. Its `REPORT.md` and coverage receipt verify 156 trained/evaluated runs and 368 figure bundles; every bundle has PNG, PDF, SVG, and CSV artifacts. This report includes the partial Ada timing evidence available when collection began; it does not establish complete A6000 timing coverage. Supplement 356618 completed the eight preprocessing figures and hardware-label annotation, bringing the exported total to 376 figure bundles. Final reporting uses the enlarged allocation and must verify complete timing coverage before the campaign can be marked complete. Corrected bounded monitor 356616 is running in `ACTIVE_REPAIR` mode; its scope and limits are described above.

The reporting code validates the exact active registry and excludes withdrawn MNIST representations from normalization, preprocessing, curves, and comparisons. It reads durable evidence, writes `outputs/tables/`, and renders matplotlib/seaborn PDF/SVG/300-dpi PNG figures and plot-data CSV files. It preserves absent cells as missing and keeps seed variability separate from repeated-batch timing variation. `REPORT.md` opens with clean/corruption and online-latency comparisons; incomplete results are explicitly marked. MNIST background sparsity prevents uniquely attributing speedups to contrast without a separate raw-empty-patch control.

## TensorBoard

Within an authorized allocation, from the project directory:

```bash
tensorboard --logdir outputs/tensorboard --host 127.0.0.1 --port 6006
```

Use an SSH tunnel to that actual allocated host, for example `ssh -N -L 6006:127.0.0.1:6006 -J sagnihot@dws-login-02.informatik.uni-mannheim.de sagnihot@ALLOCATED_NODE`, then open `http://127.0.0.1:6006`. Do not expose a public unauthenticated dashboard. `scripts/check_tensorboard_server.py` tests localhost startup and stops its own server afterwards. Training, epoch, evaluation, and latency axes have separate event paths. Every run logs exact sample-weighted metrics, normalization identities, support/execution information, bounded histograms, signed diagnostic panels, and full validation each epoch. Final scientific figures are mirrored to `study_summary`.

Optional full-data refits are disabled. After primary recipes freeze, separate full-data normalization artifacts can be fitted with `--protocol full_train_refit`; `python configure_refit.py --enable` writes separate refit configurations. These have 60,000/50,000 training images and no independent validation curve and are excluded from primary held-out aggregates. No full-data refits are launched by default.
