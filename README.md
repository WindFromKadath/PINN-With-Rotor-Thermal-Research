# Rotor Thermal Research

[中文](README.zh-CN.md)

Four reusable Python reference libraries for rotor heat transfer, neural operators and physics-informed models. The collection preserves distinct physical assumptions and model transforms for each direction. Software usability is the current acceptance criterion; historical accuracy, convergence and replacement of COMSOL have not been established by migration checks.

| Directory | Scope | Python | PyTorch |
|---|---|---|---|
| [Neural-Operator-Reference](Neural-Operator-Reference/README.md) | Neural operators, annular PINO and PI-DeepONet | 3.13.14 | 2.14.1 |
| [Pinn-Inverse-Rotor-Thickness](Pinn-Inverse-Rotor-Thickness/README.md) | Known-thickness temperature reconstruction; thickness inversion remains a research gap | 3.12.14 | 2.12.1 |
| [Pinn-Ring-Rib-2.5D](Pinn-Ring-Rib-2.5D/README.md) | Heating/cooling ring-rib temperature PINNs and coupled workflows | 3.12.14 | 2.12.1 |
| [Pinn-3D-Thermo-Mechanical-Rotor](Pinn-3D-Thermo-Mechanical-Rotor/README.md) | Transient thermal PINNs and one-way axisymmetric quasi-static thermoelasticity | 3.13.14 | 2.14.1 |

## Install and inspect

Run these commands from the collection root with uv available. Each subproject has its own lock, Python version, environment and local cache; there is no shared uv workspace. CPU and CUDA extras are mutually exclusive. For CUDA installation, follow the selected subproject's instructions.

```powershell
uv sync --directory Neural-Operator-Reference --locked --extra cpu
uv sync --directory Pinn-Inverse-Rotor-Thickness --locked --extra cpu
uv sync --directory Pinn-Ring-Rib-2.5D --locked --extra cpu
uv sync --directory Pinn-3D-Thermo-Mechanical-Rotor --locked --extra cpu

uv run --directory Neural-Operator-Reference --locked --extra cpu heatops --help
uv run --directory Pinn-Inverse-Rotor-Thickness --locked --extra cpu rotor-pinn --help
uv run --directory Pinn-Ring-Rib-2.5D --locked --extra cpu ring-rib-pinn --help
uv run --directory Pinn-3D-Thermo-Mechanical-Rotor --locked --extra cpu thermo-rotor --help
uv run --directory Pinn-3D-Thermo-Mechanical-Rotor --locked --extra cpu python ../tools/check_publication.py
```

Importing a library does not start training. Open a subproject's README for configurations, synthetic examples and data-dependent workflows. Necessary theory notes are embedded locally; historical notes are primarily in Chinese.

## Included and local assets

The Git candidate scope contains code, configurations, locks, tests, necessary curated text and sanitized metadata. Original COMSOL tables, meshes, checkpoints, selected legacy source bundles, full source documents, run artifacts, environments and caches remain local and ignored. A fresh clone therefore supports the documented basic code checks and synthetic workflows; historical model replay requires the corresponding local assets. Files in ignored folders are outside the MIT grant for this release.

[Research reading](docs/research.md) · [Publication scope and provenance](docs/publication/README.md) · [Recorded checks](docs/publication/readiness.json) · [Contribution guide](CONTRIBUTING.md)

## License

Project-authored code and original curated text in the published scope use [MIT](LICENSE), with matching license copies in all four subprojects. Installed dependencies retain their own licenses. Third-party source documents, data and weights are excluded; an index or citation does not grant redistribution rights. Upstream noncommercial DeepONet source files were excluded from the migrated implementation. Source-related limitations are recorded in each subproject.
