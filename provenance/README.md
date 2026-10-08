# Provenance

`publication_snapshot.json` records the initial 1,330 copied files and original source hashes. `sources/` contains the five byte-preserved execution source versions, distinguishing training, measurement, follow-up inference, later numerical validation, and controller orchestration. `environment/` preserves the exact environment exports.

`portability_adaptations.json` records public-package path changes separately. The original source snapshots and all scientific model/frontend computations remain unchanged. Supplemental completion receipts and selected profiler evidence have their own manifests.

`local_artifact_inventory.json` records the local raw artifact footprint. It is not a downloadable checkpoint archive or an atomic inventory of a completed follow-up campaign. See the [artifact policy](../docs/repository_layout.md) and [reproduction limits](../docs/reproducibility.md).
