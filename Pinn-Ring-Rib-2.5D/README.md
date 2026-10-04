# Pinn-Ring-Rib-2.5D

[中文与详细配置说明](README.zh-CN.md)

Heating/cooling ring-rib temperature PINNs and coupled workflows. This reusable library is one of four projects in Rotor-Thermal-Research. Migration accepts usable code/models and bounded workflows; it does not establish historical accuracy, convergence or independent scientific validation.

## Install and inspect

Run from this subproject directory. Python 3.12.14, PyTorch 2.12.1, independent uv.lock and project-local cache. CPU/CUDA extras are mutually exclusive; the CUDA extra uses the configured cu130 index.

```powershell
uv sync --locked --extra cpu
uv run --locked --extra cpu ring-rib-pinn --help
uv run --locked --extra cpu python tools/check_git_ready.py
```

Importing modules does not launch training or load original data. Full configuration and command tables are in the Chinese guide; the CLI help lists the available commands. Use explicit run IDs, retain failures and load matching model definitions, transforms and scales.

## Research and assets

[Basic clone workflows](docs/running_from_clone.md) · [Existing derivations and gaps](docs/theory/README.md) · [Publication scope](docs/publication/README.md) · [Recorded clean-copy checks](docs/publication/clone_checks.json)

Necessary curated research notes and provenance are included and are primarily Chinese. Raw COMSOL data, selected historical models and source archives, run artifacts, environments and caches stay local and ignored. Historical replay and real-data demos require these assets; a clone alone does not provide them. Hypothesis statuses remain unchanged.

## License and contributions

Project-authored code and original curated text in the published scope use [MIT](LICENSE). Dependencies retain their own licenses; third-party originals and ignored data/model archives are outside this release. Provenance indexes do not grant redistribution rights. See [Contributing](CONTRIBUTING.md).
Current relocated checks: [MOVE20261004_01](docs/publication/relocation.json).
