# 从新克隆运行 · Neural-Operator-Reference

本项目独立使用uv管理环境，Python固定为3.13.14；进入本目录后执行。四个项目的Python与PyTorch版本不同，因此各自管理`.venv`与`uv.lock`，总目录不建立共用环境。

```powershell
$env:UV_CACHE_DIR = ".cache/uv"
uv sync --locked --extra cpu
uv run --locked --extra cpu heatops --help
uv run --locked --extra cpu python -m pytest tests -q
```

1D合成示例与纯物理PIDON可自行生成输入；圆环PINO的COMSOL数据、预处理张量与历史模型不在Git中。

可运行的合成短示例使用新编号，再次运行需要改编号：

```powershell
uv run --locked --extra cpu heatops train-pino-1d --config configs/pino_1d_hybrid.yaml --smoke --run-id Clone-Demo-1D
```

## 数据与模型恢复

仓库包含代码、配置、锁文件、实现测试、必要推导整理与验收摘要。`data/raw/`、`data/processed/`、`local_reference_versions/`与`artifacts/`保存的实际内容均留在本地。恢复时保持配置约定的相对布局与单位；改路径应编辑配置，模型包需保留定义和权重配对及哈希。

每次训练使用未占用run_id；配置路径相对配置文件解析，失败保留。历史模型元数据可读[模型目录](reference_models/README.md)，基础计算与科学证据分别见[验收摘要](verification/README.md)和[理论材料](theory/README.md)。

本项目依赖安装使用包索引；研究说明、推导和验收不靠远程文档补足。完整CPU测试含历史包时应先恢复本地内容，不能把基础检查通过称作历史模型已经随Git克隆交付。
