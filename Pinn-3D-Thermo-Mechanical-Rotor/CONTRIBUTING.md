# Contribution guidelines

English | [简体中文](CONTRIBUTING.zh-CN.md)

Read README.md, MAINTAINER.md, AGENTS.md and relevant hypothesis/experiment rows first. The reference library is implemented. Install and run data-free checks with:

```powershell
uv sync --locked --extra cpu
uv run --locked --extra cpu pytest -q
```

Use a new run_id and preserve configuration, data, environment, original outputs and failures. Register a new experiment for changes to research design, physics, reference frames or metrics. Do not overwrite results or infer validated accuracy from short training. Full-budget configurations were not accepted in this first round; do not expand into long training without an explicit task.

Keep English/Chinese reader documentation synchronized. Internal maintenance notes are Chinese; the author maintains their own research understanding. Report versions, minimal reproduction and expected behavior in the current collaboration or maintenance notes, without private paths, credentials or restricted data.

Code and necessary research notes are prepared within this project; datasets, checkpoints and original archives stay local. Do not automatically commit, push, publish or change Git configuration. Use authorized materials with provenance; redistribution permissions for third-party data and weights remain unverified.

Real-data demos require the local assets described in the [clone guide](docs/running_from_clone.md).

[Return to README](README.md)

## License

Contributed project code and original text must be eligible for [MIT](LICENSE). Preserve compatible third-party attribution; data, weights and historical originals remain excluded.
