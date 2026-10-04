# Neural-Operator-Reference · 科研参考库

本项目归入 Rotor-Thermal-Research 集合，按[发布范围](docs/publication/README.zh-CN.md)准备提交；数据、权重与原件保持忽略。

新克隆请先读[安装与检查说明](docs/running_from_clone.md)；[必要理论文本](docs/theory/README.md)已随仓库保存。历史重放与真实数据示例需要另行恢复本地数据、网格及权重，下方验收和模型链接指向仓库内摘要与元数据。

本库提供可修改、可训练、可保存重载的热传导参考实现：FNO/PINO、DeepONet、PI-DeepONet、PIDON，以及圆环稳态/瞬态 PINN。采用科研项目模板，以配置和独立实验编号管理运行。

2026-10-03 迁移验收：92 项实现测试通过，6 个精选历史模型和 16 组训练产物成功严格重载，原文件与迁移副本通过 SHA-256 核验。目标是代码与模型可用；短训练误差较大，未据此确认精度、收敛性或替代 COMSOL 的能力。

## 来源与范围

完整来源是本地 `01_算子学习_已弃` 外层目录，包括 `PythonAgnetCoding`、`DeepONet` 和 `neural-operator-exploration`。后者只是归档索引，并非完整数据来源。原目录保持原样。

- 模块化实现位于 `src/heatops/`，导入不会加载数据或开始训练。
- 50 份 CSV 已复制到 `data/raw/comsol_annulus/`；完整处理数据 `data/processed/annulus_polar_64.pt` 形状为 `(50,100,64,64)`，短示例为 `(4,10,16,16)`。
- 4 组历史参考包位于 `local_reference_versions/`，包含匹配源码、原权重、转换权重、网络定义和输入/输出变换。
- `source_inventory.json` 记录完整来源 661 份文件的相对路径、大小和哈希；`copied_files.json` 记录 73 份实际迁移文件。
- 上游 `deeponet` 的逐字拷贝文件 `utils.py`、`ADR_solver.py` 没有迁入实现库；其他未选分支保留来源清单。

数据、权重、历史快照和产物默认被 Git 忽略。本轮没有提交、推送、安装 Hook 或更改 Git 配置。

## 环境与运行

以下命令从项目根目录执行，适用于 PowerShell。Python 固定为本机已验证的 **3.13.14**；`pyproject.toml` 声明依赖，`uv.lock` 锁定，缓存位于 `.cache/uv`。

```powershell
uv sync --locked --extra cpu
uv run --locked --extra cpu heatops --help
uv run --locked --extra cpu python -m pytest tests -q
```

已验证 CPU 环境：PyTorch 2.14.1+cpu、DeepXDE 1.15.0、neuraloperator 2.0.0。`cpu` 与 `cuda` extras 互斥；CUDA 使用官方 cu130 索引，已锁定但本轮未验证运行。`requirements-lock.txt` 是 uv 导出的兼容清单，uv.lock 为环境依据。

### 数据与短示例

两份处理数据已生成。预处理拒绝覆盖已有输出；重新转换须在配置中指定新输出路径。

```powershell
uv run --locked --extra cpu heatops preprocess --config configs/preprocess_annulus.yaml --smoke
uv run --locked --extra cpu heatops preprocess --config configs/preprocess_annulus.yaml
```

训练统一接受 `--config`、`--smoke`、`--run-id`。下面使用新编号；再次运行须换编号。

```powershell
uv run --locked --extra cpu heatops train-pino-1d --config configs/pino_1d_hybrid.yaml --smoke --run-id Demo-1D
uv run --locked --extra cpu heatops train-pino-annulus --config configs/pino_annulus_polar.yaml --smoke --run-id Demo-Annulus
uv run --locked --extra cpu heatops train-pidon --config configs/pidon_annulus.yaml --smoke --run-id Demo-Pidon
```

| 命令 | 配置与作用 |
|---|---|
| `train-pino-1d` | `pino_1d_fno.yaml`、`pino_1d_physics.yaml`、`pino_1d_hybrid.yaml`：三种损失组合 |
| `train-pino-annulus` | `pino_annulus_polar.yaml`：圆环数据/物理约束算子 |
| `train-pidon` | `pidon_annulus.yaml`：以热流函数为输入的纯物理算子 |
| `train-pi-deeponet` | `pi_deeponet_1d.yaml`、`pi_deeponet_2d.yaml`：源项算子 |
| `train-deeponet` | `deeponet_2d.yaml`：FDM 数据监督 |
| `train-pinn` | `pinn_annulus_steady.yaml`、`pinn_annulus_transient.yaml`：圆环对照，按模式建子目录 |
| `compare` | 汇总指定的三组 1D 运行；参数见 `heatops compare --help` |
| `evaluate` | 严格重建指定运行的网络并重放原评价点集 |
| `legacy-evaluate` | 重放历史包，保存预测、指标和图 |

数据路径相对配置文件解析，产物写入该项目 `artifacts/<run_id>/`。使用绝对配置路径可从其他启动位置运行。实际 smoke 覆写保存在 `config.json`；FNO 循环包含第 0 步，`epochs=5` 对应 6 次更新。完整配置保留较大预算，本轮未执行。

### 模型预测与重载

```powershell
uv run --locked --extra cpu heatops evaluate --run-dir artifacts/E116 --out-dir artifacts/Replay-E116
uv run --locked --extra cpu heatops legacy-evaluate --package local_reference_versions/Fno-1D-Comparison --model hybrid --out-dir artifacts/Legacy-Hybrid
uv run --locked --extra cpu python tools/verify_reference.py --out-dir artifacts/Verify-Local
```

评估输出目录须不存在；评估不重训、不覆盖训练指标。训练保存配置、环境、种子、源码哈希、适用的数据哈希、日志、权重、预测、指标和图；失败保存异常记录。所有权重严格加载，历史包另核验哈希与格式版本。

## 数据与模型契约

COMSOL 温度为 °C，热流为 W/m²，时间为 s，长度转换为 m，角度为 rad。真实 CSV 时刻为 0–99 秒；没有时间列头时须显式提供时刻，单位不符报错。极坐标张量为 `(sample,time,r,theta)`，角度含 0 与 2π；有效掩码参与插值，避免域外零值污染边缘温度，但插值仍是近似。

新圆环 PINO 使用 `T*=(T-T_inf)/T_scale`、`q*=q/q_base`、`r*=r/R_out`、`t*=t/t_max`。已有 t=0 不重复插入，缺少才补零初始帧。合成数据按配置生成并缓存；数据监督训练使用独立种子划分，纯物理算子训练/诊断分别抽取函数样本。

历史权重不能直接替换新配置权重。`heatops.models.legacy.load_legacy_model` 返回网络、规格和权重记录，调用方须采用规格中的 `transforms`；CLI 已执行对应变换。

## 本地精选版本

| 参考包 | 保留模型 | 选择依据与边界 |
|---|---|---|
| `Fno-1D-Comparison` | data / physics / hybrid 各一组 | 各自 final/checkpoint 在固定 4 个原数据样本上比较；原划分未知，仅作候选诊断 |
| `Pino-Annulus-Polar` | 极坐标 optimized | 唯一匹配极坐标定义的候选，严格加载及有限预测通过 |
| `Pidon-Annulus` | open-15000 | 5k/10k/15k 在固定径向剖面上与 C-N 参照比较 |
| `Pinn-Annulus-Transient` | open-15000 | 匹配 open/polar2 候选作同一小规模诊断 |

具体得分、来源、网络与变换见各包 `model_spec.json`。这是限定候选中的选择，不宣称全局最优或泛化能力；原问题随历史源码保留。

## 验证与实验索引

实际设置与结果见 [experiments.csv](experiments.csv)，科学假设见 [hypotheses.csv](hypotheses.csv)。`completed` 仅表示执行完成；实现测试与科学判断分开记录。

- E106/E107/E110/E111 的历史 0.1.0 产物已补登记并重载核查，原指标保留。
- E108/E109 补做圆环 PINO 与修正后 PIDON；E112 为修正后 1D 混合 PINO。
- E113a/E113b 同种子短训损失与预测逐点一致，范围仅限本机 CPU 短训练。
- E114 重放 6 个历史模型和 15 组产物；E116 完成最终周期边界修正后的圆环短训及重载。
- E115 记录最终 92 项实现测试、文件核验和入口检查。总验收见 [verification.json](docs/verification/records/migration_20261003/verification.json)。最初真实性审计保留在 `artifacts/audit_20261003_01/`，反映修正前状态。

H101/H102 有解析场及求解器测试，H104 有短训证据；假设状态留待负责人判断。H102 的点式 autograd 检查不能证明非局部 FNO 的坐标 VJP 正确。H103 完整排序与精度复现未执行。

## 已采纳的关键取舍

| 编号 | 决定与理由 |
|---|---|
| D101 | 模块化重写，原目录只读，采用科研模板和独立编号 |
| D102 | 不迁入上游 deeponet 逐字拷贝实现；公开前另核查许可 |
| D103 | 新实现换算集中在 `problems/`、`data/` 和配置；历史包绑定原变换 |
| D104 | 新圆环 PINO 使用极坐标，其他分支建清单 |
| D105 | 显式种子、划分与严格重载，拒绝复用运行目录 |
| D106 | 修正 PIDON 热流 aux_var 输入、周期边界筛选及重复加权的总损失 |
| D107 | FNO 使用输出场有限差分；非局部 FFT 模型的坐标总和 VJP 不是逐点物理导数。解析场 autograd API 保留；周期 BC 独立检查两端斜率 |
| D108 | 读取真实时间列头，避免重复初始帧，使用有效掩码插值，单位错误明确失败 |
| D109 | 按用户验收范围保留成熟可用历史模型，完整精度复现另设实验 |

## 作者视角与协作入口

作者的研究理解与疑点由作者自行维护，本轮没有代写。协作规则见 [AGENTS.md](AGENTS.md)。接手先读配置、实验行与原始结果，不将迁移验收直接转写为科研假设获支持。

代码与必要说明现按本项目提交范围整理；本地数据、权重和历史原件继续忽略。实现源码及入口文档不嵌入真实姓名或本机绝对路径。



## 研究材料整理

[模型、现有推导与缺口](docs/theory/README.md)汇合代码目录、毕业设计、历史备份和文献中的相关材料，保留仓库内整理稿、来源编号与哈希。本轮只整理既有内容并列出缺口，未补写证明、重新训练或调整假设支持状态。详见本项目资料入口。

## 开源许可与集合布局

[English](README.md) · [MIT](LICENSE) · [贡献说明](CONTRIBUTING.zh-CN.md)

公开范围内的项目自有代码与原创整理稿采用 MIT；依赖保留原许可，历史数据、模型与第三方原件不纳入本次发布授权。迁移后的 uv 环境按原锁文件重建，科学假设状态不变。
移动后实际检查：[MOVE20261004_01](docs/publication/relocation.json)。
