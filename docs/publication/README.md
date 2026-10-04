# Publication scope and provenance

[中文](README.zh-CN.md)

The four subprojects were moved into this collection without importing any Git history. Their own code, locks, tests and original curated text are the intended public content, licensed under MIT with identical LICENSE files at the collection and subproject roots. The outer parent folder and its unrelated fifth project stay outside this scope.

Each project's source manifests bind selected historical material to relative paths and hashes. Those manifests provide provenance, not a transfer of rights. The neural-operator implementation explicitly excludes upstream noncommercial DeepONet utils.py and ADR_solver.py. Dependencies are installed through uv and retain their own licenses; their implementation files are not vendored here. Historical source archives, papers, COMSOL tables, meshes and weights remain excluded. Do not describe all dependencies as MIT or apply this license to third-party originals.

[Neural-operator provenance](../../Neural-Operator-Reference/README.zh-CN.md) · [Inverse provenance](../../Pinn-Inverse-Rotor-Thickness/README.zh-CN.md) · [Ring-rib provenance](../../Pinn-Ring-Rib-2.5D/README.zh-CN.md) · [Thermal-mechanical provenance](../../Pinn-3D-Thermo-Mechanical-Rotor/README.md)

## Checks and limits

[Current candidate check](readiness.json) and [candidate hashes](manifest.json) cover public text/code. Local relocation evidence is kept in ignored per-project artifacts. Original configurations, scientific source, tests and uv locks remain unchanged. CPU environments are rebuilt at the new paths; prior environment folders are retained and ignored. Current CPU/model results are recorded in the readiness report, separately from earlier scientific experiments.

No Git repository, remote, commit, push, hook or identity change is created by this task. Before actual publication, review staged bytes, author identity, intended branches/history and attachments using the general MIT/publication workflow maintained in the separate collaboration-template vault. This repository's commands and necessary theory do not depend on access to that vault. Candidate checks do not replace a dedicated secret scanner, image inspection or rights review of future additions.
