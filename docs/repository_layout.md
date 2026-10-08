# Repository layout and artifact policy

The public repository organizes the project without moving or modifying the live experiment roots. `sparse_contrast_benchmark_large_batch/`, `training_dense_testing_sparse/`, and `sparse_contrast_benchmark/` remain the operational Ceph workspaces and are excluded by `.gitignore`. Their path-bound jobs continue to use immutable source snapshots and the same output identities.

| Public directory | Purpose and authority |
|---|---|
| `src/sparse_contrast/` | Readable main implementation, based on the frozen measurement source. Only documented package-path adaptation differs. |
| `src/dense_sparse/` | Inference intervention adapter and archived follow-up control flow; public import paths are adapted, scientific inference is preserved. |
| `experiments/training_sparse_testing_sparse/` | Primary larger-batch protocol, registry, configs, normalization artifacts, original entry scripts and checks. |
| `experiments/training_dense_testing_sparse/` | Follow-up case manifest and original protocol. Absolute historical paths remain evidence. |
| `experiments/legacy_batch8/` | Preserved earlier training recipe/source. Never combine these rows with the primary study. |
| `provenance/sources/<hash>/` | Byte-preserved immutable training, measurement, follow-up inference, numerical-validation repair, and controller-repair versions. These are the original execution authorities. |
| `provenance/environment/` | Exact exported environment and Python package lock. |
| `results/<study>/` | Exported available scientific tables. Empty tables and partial coverage are preserved honestly. |
| `results/diagnostics/` | Selected saved profiler/cell evidence with source hashes, not newly measured profiles. |
| `plots/` | Exactly three top-level comparison PDFs, companion CSVs, and indexed historical plot directories. |
| `analysis/` | Portable reconstruction of the requested comparisons from public tables. |
| `analysis/original_exporters/` | Original source of the historical publication exporters; these retain cluster paths and are not portable entry points. |
| `debugging/scratchpad.md` | Requested operational history: decisions, failures, repairs, known gaps, and current continuation state. |

## Publication evidence

[publication_snapshot.json](../provenance/publication_snapshot.json) binds each initial exported source file to its original path, target path, size, and SHA-256. It also records source identities and explicitly omitted tables. The export ran in CPU allocation 361154 and completed successfully. Later public path adaptations and supplemental completion/diagnostic receipts are separately manifested. An adapted public convenience file must never be attributed to an unchanged original-source digest.

[local_artifact_inventory.json](../provenance/local_artifact_inventory.json) inventories the live output trees by metadata, including their sizes, suffix counts, and files too large for ordinary Git. Its counts are a dated scan while timing continued, not an atomic final campaign snapshot. Data symlinks were not followed. External dataset storage is therefore not included in its output totals.

Raw checkpoints, per-sample predictions, exhaustive timing traces/arrays, TensorBoard event streams, and operational logs remain in the original local roots. Checkpoint completion receipts and registered case/configuration hashes identify the exact trained state. Public tables and selected diagnostic evidence support the published comparisons; they do not replace a full raw-artifact archive for independent inference reproduction.

The GitHub repository is public as explicitly requested. Credentials, private keys, local environment files, datasets, and model blobs are excluded. No LFS upload or external artifact-hosting service has been provisioned. Upstream source attribution is preserved in [NOTICE.md](../NOTICE.md); no license was invented.

## Read current documents before archived instructions

The root README and `docs/` describe the final public layout and current scientific interpretation. Copied experiment READMEs, REPORTs, scheduler scripts, and source snapshots preserve historical context, including old relative `outputs/` paths, original Ceph locations, and superseded intermediate status. Those are archival evidence, not instructions to overwrite a completed run or launch a second controller.

On the original cluster, current machine-owned state remains `training_dense_testing_sparse/STATUS.json` and `outputs/campaign.json`. The existing controller is authoritative. Its parent snapshot can reflect the state when follow-up was released; use the primary study's verified coverage to establish primary completion. Public Git files are snapshots and do not automatically update as jobs finish.
