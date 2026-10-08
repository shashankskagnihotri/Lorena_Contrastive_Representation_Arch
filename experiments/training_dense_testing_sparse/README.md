# training_dense_testing_sparse

Evaluation-only follow-up to the existing large-batch `training_sparse_testing_sparse` campaign. The exact scientific protocol is in [PLAN.md](PLAN.md), the full checkpoint/intervention/work matrix in `study_manifest.json`, and live state in `STATUS.json`. The shared project record is [debugging/scratchpad.md](../debugging/scratchpad.md).

All learned weights and normalization come from the original 60 final checkpoints trained with zero imposed sparsity. Dense-trained and compact-at-zero-trained families stay distinct. The inference sweep is0,10,...,90%, with0% an execution control. The provisional raw-pixel sweep is labelled separately from contrast pruning and has unchanged raw dense references. MNIST has no color conditions. No training or fitting is performed here.

The user's October5 instruction authorizes immediate evaluation submission while parent final reporting continues. The recorded scheduling amendment releases the existing controller from its original parent-completion gate after the source-checkpoint audit; see `outputs/checks/evaluation_release_2026-10-05.json`. Accuracy uses the original full official data/splits and selected evaluation batches. Latency uses the original isolated timing protocol and a separate homogeneous hardware cohort. New output paths cannot overwrite parent experiments.

Use the pinned existing environment:

```bash
cd /ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/training_dense_testing_sparse
BENCH_PY=/ceph/sagnihot/miniconda3/envs/sparse-contrast-bench/bin/python
"$BENCH_PY" -m dense_sparse.controller status
```

Launch is idempotent and submits a scheduler-hosted owner; deployment first freezes and validates source. Do not rerun setup against a changed source after study creation. The source used by all jobs is the immutable directory recorded in the manifest.

```bash
"$BENCH_PY" -m dense_sparse.controller launch
"$BENCH_PY" -m dense_sparse.controller launch-monitor
```

Only when a stop or resume is intended:

```bash
"$BENCH_PY" -m dense_sparse.controller stop
"$BENCH_PY" -m dense_sparse.controller resume
```

Stop creates this study's STOP marker and cancels positively identified study jobs. Parent STOP also prevents new launches. Resume reconciles cancelled worker ownership and preserves committed cells, attempts and retry counters. A separate `outputs/agent_monitor/STOP` stops only repair monitoring; it is not cleared implicitly. The monitor's96-hour deadline and8-call limit survive restarts and include time waiting for the parent. Expired limits are reported, never silently reset.

Worker/report entrypoints are implemented for controller use inside allocations:

```bash
python -m dense_sparse.worker evaluate --case /absolute/path/to/config.json
python -m dense_sparse.worker latency --case /absolute/path/to/config.json --execution compact --batch-size 1
python -m dense_sparse.report
```

Do not run substantial evaluation/reporting on login nodes. The October5 user-authorized scheduling amendment supersedes the original parent gate; it does not claim parent reporting is complete. Production commands are saved in `outputs/submissions/` with exact job IDs and immutable source paths. `outputs/slurm/` contains scheduler logs. `outputs/checks/` contains allocated validation/checkpoint-audit receipts.

Per-case outputs live in `outputs/cases/<dataset>/<architecture>/<representation>/trained_<execution>/infer_<execution>/p<percent>/seed<seed>/<test_hash>/`. They include official per-cell prediction/token evidence, evaluation summaries, diagnostic panels and timing worker directories. `outputs/tensorboard/` keeps independent inference identities and aggregate figures. `outputs/tables/` and `outputs/figures/` are derived from saved evidence; `REPORT.md` is written after complete accuracy and refreshed after complete latency. Incomplete coverage is explicit.

The bounded repair monitor checks progress every5minutes and can invoke the previously verified authorized Codex runtime for actual incidents. It cannot post unsolicited messages to the original chat. Its live identity, heartbeat, limits, incidents and invocation evidence are in `outputs/agent_monitor/`. Read [MONITOR.md](MONITOR.md) before any repair.
