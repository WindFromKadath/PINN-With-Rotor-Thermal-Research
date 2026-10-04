# Pinn-3D-Thermo-Mechanical-Rotor

This project belongs to the Rotor-Thermal-Research collection; see the [publication scope](docs/publication/README.md). Data/checkpoints stay excluded.

See [clone guide](docs/running_from_clone.md) for installation and data-free checks, and [in-repository theory notes](docs/theory/README.md) for model definitions. Historical replay and real-data demos require the excluded local datasets, meshes and checkpoints. Verification links below point to stored summaries, not downloadable weights.

English | [简体中文](README.zh-CN.md)

A standalone research reference library for transient 3D heat conduction, one-way thermal-to-mechanical coupling, axisymmetric thermoelasticity, data-only MLP baselines, IWG reference frames, and steady three-domain X-PINNs.

Migration completed on 2026-10-04 for software, model and short-workflow usability. Full-budget accuracy reproduction was not performed. Original sources and the research template remain unchanged. All data, weights and outputs stay local.

## Scope and provenance

The research template supplies bilingual reader documentation, maintenance notes, collaboration rules and hypothesis/experiment indexes. Complete sources include the pinn-3d-thermo-mechanical-rotor snapshot and its sibling LittleTinyTest/FXXKINGGOD data and weights.

Thermal inputs are `[r/L,z/L,t/tau,sin(theta),cos(theta)]`, with output `(T-T0)/dT`. Mechanical inputs are `[r/L,z/L,t/tau]`, with outputs `[u_r/U,u_z/U]` and `U=alpha*dT*L`. Mechanics solves quasi-static equilibrium under a changing thermal load, without inertia or azimuthal displacement. The bridge freezes thermal parameters while preserving input derivatives; its default angle is theta=0.

The main and formal `ring.stl` meshes cover the lower ring, approximately z=-34.396 to -1.396 mm. A larger configuration bounding box does not imply full-rotor training. The steady X-PINN workflow uses separate lower-ring, rib and upper-ring meshes.

| Profile | q / W·m⁻² | h / W·m⁻²·K⁻¹ | dT / K | tau / s |
|---|---:|---:|---:|---:|
| little_transient, fxx_transient | 100000 | 20 | 300 | 200 |
| formal_thermal, xpinn_steady | 50000 | 50 | 100 | 100 |
| iwg | Radial 400→300 K field | Not applicable | 106.85 | 200 |

Each package binds its architecture, scales, output transform and reference frame. Identical filenames do not establish compatibility. NTK/ReLoBRaLo, MoE, SDF and other explorations remain indexed source snapshots; their full training comparisons were not rewritten.

## Installation and commands

Python is pinned to **3.13.14**, with dependencies locked by uv.lock. Mutually exclusive CPU/CUDA extras use the official PyTorch CPU/cu130 indexes. The cache lives in `.cache/uv`.

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location).Path '.cache/uv'
uv sync --locked --extra cpu
uv run --locked --extra cpu thermo-rotor --help
uv run --locked --extra cpu thermo-rotor demo --config configs/demos/coupled.toml
uv run --locked --extra cpu pytest -q
```

Run these commands from the project root. Default CLI configurations resolve from the installed project; configured paths resolve relative to the configuration file. Other working directories can use `uv run --project <project-directory> --locked --extra cpu ...`.

| Task | Command following `uv run --locked --extra cpu` |
|---|---|
| Coupled short workflow | `thermo-rotor demo --config configs/demos/coupled.toml` |
| Thermal training | `thermo-rotor train-thermal --config configs/demos/thermal.toml` |
| Mechanical training | `thermo-rotor train-mechanical --config configs/demos/mechanical.toml` |
| Two smaller MLP label budgets | `thermo-rotor train-mlp --config configs/demos/mlp.toml` |
| IWG constraint/projection | `thermo-rotor train-iwg --config configs/demos/iwg.toml` |
| Steady X-PINN stages A–D | `thermo-rotor train-xpinn --config configs/demos/xpinn_steady.toml` |
| Separate formal thermal case | `thermo-rotor demo --config configs/demos/formal_thermal.toml` |
| COMSOL wide-to-long conversion | `thermo-rotor preprocess --config configs/preprocess.toml` |
| Interior and boundary sampling | `thermo-rotor sample --config configs/demos/coupled.toml` |
| Saved-run evaluation | `thermo-rotor evaluate --run-dir artifacts/R102` |
| Rebuild plots | `thermo-rotor plot --run-dir artifacts/R102` |
| Predict using a saved run | `thermo-rotor predict --run-dir artifacts/R102 --model thermal --input artifacts/R102/thermal_inputs.csv --output artifacts/new_prediction.csv` |
| Historical main model replay | `thermo-rotor legacy-evaluate --bundle little` |
| Gather run metrics | `thermo-rotor compare --run-dirs artifacts/R102 artifacts/R108 --output artifacts/new_comparison.json` |

Run IDs are generated automatically unless supplied. Existing IDs and prediction/conversion outputs are rejected. Failures retain configuration and failure.txt. The standalone mechanical example binds `artifacts/R102/thermal.pt`; change `paths.thermal_checkpoint` to use your own thermal run.

CUDA checks use a separate `.venv-cuda`, preserving the CPU environment. For a new GPU run, set `$env:UV_PROJECT_ENVIRONMENT='.venv-cuda'` in a separate PowerShell process, then run `uv sync --locked --extra cuda --no-dev` and `uv run --locked --extra cuda --no-dev thermo-rotor demo --config configs/demos/coupled.toml --device cuda`.

Demo configurations use seed=42, 40 Adam steps for each thermal/mechanical network, 128 interior points, 64 points per boundary, 64 initial-condition points, three evaluation/anchor times and up to 32 training anchors per time. MLP demos use 32/96 total labels. Historical 1100/33000 models both have approximately 66,818 parameters. IWG and X-PINN demos use 20 steps per stage, with L-BFGS disabled; tests separately cover a small fixed-closure L-BFGS path.

Full configurations in `configs/` retain the principal source budgets and were not executed. Original YAML files are archived. Reference training uses explicit fixed small batches; complete source sampling/optimizer accuracy reproduction has not been validated. The short IWG third stage uses Adam to exercise the full loss, whereas the full configuration uses L-BFGS.

## Data and model contracts

Internal fields use SI: x/y/z/r in m, theta in rad, t in s, T in K, u_r/u_z in m and stress in Pa. Importers support COMSOL `expression (unit) @ t=...` wide tables and processed long tables. Length, displacement, temperature and time units must be explicit. Negative time, nonpositive radius/temperature, nonfinite values, missing columns, inconsistent units and incomplete displacement pairs are rejected. T_true maps to T without modifying Kelvin temperatures.

CSV ingestion is chunked. Anchors use bounded seeded priority sampling independently per time, with the actual selected points saved. This is a new reference sampling protocol, without claiming identical rows to the historical reservoir implementation. Different training/evaluation seeds do not establish leakage-free historical splits.

The default preprocessing configuration exercises the first 32 spatial rows of the real COMSOL table. Full conversion uses `configs/preprocess_full.toml` and may generate several GB; it was not executed in this round.

Required tables, STL files and selected models are local. The approximately 5 GB ElsMisesSCheck.csv remains hash-indexed; full stress-reference error reproduction was not performed. Temperature/displacement metrics use saved evaluation inputs. Formal thermal and steady X-PINN runs lack matching COMSOL references: their T0 diagnostic metrics and plots only exercise the workflow.

```python
from thermo_rotor.models.checkpoint import load
from thermo_rotor.evaluation.predict import predict

model, profile, metadata = load("artifacts/R102/thermal.pt", expected_kind="thermal")
# frame contains SI r,z,t,theta columns.
values_K = predict(model, frame, profile, metadata["spec"], metadata["gauge"])
```

Imports do not read data, load weights, create directories or train. Checkpoints consist of a tensor `.pt` file and a matching `.json` definition. Move both files and saved evaluation inputs together. Loading checks format, strict architecture, finite weights, SHA-256 and caller-specified profiles. Evaluation uses the selected run's models and snapshots without reading original source paths.

## Historical packages and evidence

See [index.json](docs/reference_models/index.json) and [source_manifest.csv](docs/reference_models/source_manifest.csv). Eight packages contain ten models, keeping original `.pth` files alongside portable `.pt/.json` files.

| Package | Purpose and selection |
|---|---|
| little | Preferred paired thermal/mechanical main models; zero same-weight prediction difference |
| fxx | Matched thermal/mechanical source comparison |
| mlp_1100, mlp_33k | Historical label-budget baselines; mlp_33k preferred on the diagnostic sample |
| iwg_fxx_constraint | Preferred constraint candidate; restores the source anchor and improves bounded diagnostics over the Little evaluator |
| iwg_constraint, iwg_projection | Preserved Little evaluator semantics and projection comparison |
| fxx_ntk | Usable historical NTK contrast; worse bounded diagnostics, explicitly not preferred |

Selection is limited to matching cases, source best checkpoints and fixed small samples, without a global-best claim. The Little constraint evaluator omitted the trained anchor; replay can therefore show large error. The FXX candidate restores its explicit source anchor. Projection saves one global axial shift with its reference points, making later inference independent of batch size. This is not a physical correction or evidence from linear-fit alignment.

- [R101](docs/verification/records/R101/verification.json): nine initially selected finite models; zero temperature difference and maximum displacement difference approximately 1.1e-11 m; main PDE compatibility passes. [R120](docs/verification/records/R120/metrics.json) adds the tenth preferred constraint model.
- R102–R106: coupled, MLP, IWG, steady X-PINN and formal thermal short training, checkpoint reload, evaluation and plots completed. IWG short-budget losses remain large and fluctuate; convergence is not claimed.
- The initial R105 port omitted the source 380 K temperature anchor. It was restored and checked in R125, the delivered steady X-PINN reference; the earlier result remains preserved.
- [R107](docs/verification/records/R107/verification.json): all 774 copied files match hashes; portable CLI execution, relocated runs and overwrite protection pass; eight historical packages replay locally.
- R102/R108 CPU runs have identical seed-42 losses and prediction CSV hashes. Implementation tests cover derivatives, analytic fields, frozen bridges, interface gradients, units, geometry, reload and failure cases.
- Final CPU/GPU checks and exact environments appear in the [delivery report](docs/verification/records/migration_20261004/verification.json); runs are registered in [experiments.csv](experiments.csv).

## Limitations, maintenance and rights

Source assumptions retain quasi-static mechanics, theta=0 bridging, constant properties, normal flips and the original axial-stretch target. Its scale concern requires a separate experiment. X-PINN differentiates source flux before detaching; stage D retains lower→rib→upper interface gradients rather than symmetric two-way losses.

Software migration does not validate full PINN accuracy, convergence, material applicability or replacement of COMSOL. Scientific hypothesis statuses remain under the author's review; implementation success does not mark them supported.

See [MAINTAINER.md](MAINTAINER.md), [AGENTS.md](AGENTS.md) and [contribution guidelines](CONTRIBUTING.md). The author maintains their own research understanding. Public redistribution permissions for source materials, data and weights have not been established. Large files and private snapshots are ignored; this subproject is part of the collection submission scope. No commits, pushes or Git configuration changes were made.


## Research materials

[Existing derivations and research gaps](docs/theory/README.md) organize local source materials by model, scale, version and evidence. This review adds no new proofs or training, and does not change scientific hypothesis status. Necessary reading notes, source hashes and evidence summaries are included in this repository; originals remain local.

## License

Project-authored code and original curated text in the published scope use [MIT](LICENSE). Installed dependencies retain their own licenses; ignored data, weights and third-party originals are outside this release.
Current relocated checks: [MOVE20261004_01](docs/publication/relocation.json).
