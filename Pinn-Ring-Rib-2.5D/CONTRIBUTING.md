# Contributing

[中文](CONTRIBUTING.zh-CN.md)

Describe the affected configuration and expected behavior, and include a minimal reproducible example using synthetic or permitted data. Remove private paths, credentials and original confidential logs from issue reports. If a GitHub remote is later configured, use its issues and pull requests; none is required for local development.

Use the affected subproject's locked uv environment. Run relevant tests from that subproject; historical-model tests require local assets and must not be reported as passed when unavailable. The following checks are available in this document's directory:

```powershell
uv run --locked --extra cpu python -m pytest -q
uv run --locked --extra cpu python tools/check_git_ready.py
```

Keep scientific changes separate from relocation or publication edits. Record new experiments with unique run IDs, explicit conditions and retained failures. Synchronize public English/Chinese commands, interfaces, versions and limitations; original research notes may remain Chinese.

Contributed project code and original text use [MIT](LICENSE). Submit only material you may license under MIT, with attribution for compatible reused material. Installed dependencies keep their own terms. Do not include original data, third-party papers, historical source archives, checkpoints, environments or audit logs. Review actual staged changes and author identity before committing, then history and attachments before publication; these commands do not approve a push.
