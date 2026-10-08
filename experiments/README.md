# Experiment specifications

- **Primary:** [training_sparse_testing_sparse](training_sparse_testing_sparse/) is the completed larger-batch study, physically executed in `sparse_contrast_benchmark_large_batch/` on the original filesystem.
- **Follow-up:** [training_dense_testing_sparse](training_dense_testing_sparse/) freezes the eligible zero-imposed-sparsity checkpoints and changes inference support; accuracy evaluation is complete, timing remains partial at publication.
- **Historical:** [legacy_batch8](legacy_batch8/) preserves the earlier batch-8 recipe and evidence. It is not the main study.

These directories preserve original configuration hashes and absolute provenance. They include archival scripts and original checks; they do not include datasets or checkpoints. Some historical README/REPORT links target local `outputs/` paths, which are intentionally not replicated as large public directories. Public results are in `results/` and plots in `plots/`.

Use the current [protocol](../docs/experiment_protocol.md), [method](../docs/method.md), and [reproduction guide](../docs/reproducibility.md) before attempting execution. Do not launch an archived scheduler controller against the active campaign. New scientific experiments require a new source/configuration identity rather than editing a completed registry in place.
