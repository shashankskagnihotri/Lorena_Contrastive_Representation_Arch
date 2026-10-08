# Selected performance evidence

This directory supports [the detailed performance analysis](../../docs/performance_analysis.md). It contains saved observations, not new model runs or a replacement timing campaign.

The deterministic selection is seed 0, official clean test inputs, raw dense and grayscale compact at 60%, both architectures and datasets, and batches 1 and 8. There are 16 selected cells. Selection was by these identities, not favorable accuracy, timing, or profile outcomes. The 200 measured support records per cell are retained; the single instrumented profile is the first measured input batch. Three-seed headline latency comes from the full [parent latency overview](../training_sparse_testing_sparse/latency_overview.csv), not these seed-0 profiles.

`source_sha256_manifest.json` records exact byte copies, original project-relative source paths, SHA256 and byte size. The archive contains **91 unique copied files, 19,498,580 bytes**, plus derived tables, selection/config metadata and this reproduction script. Multiple manifest entries may map to the same deduplicated panel file; copying verified that their bytes were identical. The four unique clean panels cover two datasets × two batch sizes.

Each selected cell includes:

- `summary.json`: original medians, scopes, protocol, isolation and measurement identities.
- `profile.json`: original instrumented event aggregates, matched uninstrumented call and profiling-overhead ratio. Inclusive nested intervals are not additive stage latencies.
- `tokens.jsonl`: all 200 original native-support and executed-token records, with batch sample IDs.
- `hardware.json`: exact measurement-session metadata, including CPU/GPU class, versions, threads and UUID.
- `zero_input_diagnostic.json`: original zero-input reference diagnostic, separate from official accuracy and latency.

The selected panel membership files are under `panels/`. Full-training-partition support CSVs are under `preprocessing/`; they describe clean training inputs before augmentation, not inference latency. The normalization table contains original fit values and guard flags. Cross-study accuracy tables preserve all source-family distinctions. `selected_execution_controls.csv` extracts the 48 compact/dense-masked clean gray-60 timing rows across three seeds, four dataset/architecture groups and batches 1/8. Its adjacent provenance JSON records the source table hash and verifies exact checkpoint/normalization identity for each pair. This extracted table is additional to the 91 byte-copied files.

`selected_training_configs.json` contains the eight seed-0 original configurations without changing their configuration fields or hashes. `selected_final_training_epochs.csv` extracts all 24 final raw/gray60 records, three seeds × four dataset/architecture groups × two representations. It includes each original epoch-file SHA256 and confirms the requested 50/200-epoch budget; zero-indexed final epochs are 49/199. Online training predictions are accumulated while weights change in train mode. Validation is a separate fixed-weight evaluation of the held-out validation partition.

## Reproduce the diagnostic tables

From the repository root:

```bash
python results/diagnostics/reproduce_summary.py
```

Only the Python standard library is required. The script first verifies all 91 copied-file SHA256 values, then regenerates:

- `support_summary.csv`: per-image mean native support, executed sequence length, padding, stage counts and summed window-length groups.
- `profile_summary.csv`: selected one-call instrumentation counts and inclusive CPU intervals.
- `arithmetic_proxy.csv`: multiply-add estimates using actual executed shapes for each saved batch, averaged afterward.

The arithmetic table counts one multiply-add pair as one MAC, approximately two arithmetic FLOPs. It covers QKV, attention products, output projection and MLP; Swin merges are listed separately. It excludes stem/head, normalization, nonlinearities, indexing, memory traffic, launch costs and telemetry. It is not a measured FLOP counter or a runtime predictor.

ViT's block estimate is `12 * (12*L*D**2 + 2*L**2*D)` per image for 12 blocks, with `D=384` and each actual batch's maximum executed length `L`, including its class token. Swin's block estimate is `12*K*D**2 + 2*D*attention_pairs`, summed over actual blocks, divided by batch size. Each merge contributes `8*D**2` per active output parent. These formulas count executed padding and full dense attention arrays where the implementation computes them.

The archive intentionally omits large Chrome traces, raw timing streams and checkpoint tensors. Original summaries retain references and hashes for some omitted files; the public subset is sufficient for the calculations above, but not for revalidating every full-study completion receipt. Some exact source JSON/CSV retains original machine paths as historical provenance. Those paths are not dependencies of this reproduction command. No model is loaded, no GPU is initialized, and no timing is performed.
