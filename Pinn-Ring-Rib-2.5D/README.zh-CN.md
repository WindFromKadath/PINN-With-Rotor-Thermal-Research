# Pinn-Ring-Rib-2.5D · 科研项目

本项目归入 Rotor-Thermal-Research 集合，按[发布范围](docs/publication/README.zh-CN.md)准备提交；数据、权重与原件保持忽略。

新克隆请先读[安装与检查说明](docs/running_from_clone.md)；[必要理论文本](docs/theory/README.md)已随仓库保存。历史重放与真实数据示例需要另行恢复本地数据、网格及权重，下方验收和模型链接指向仓库内摘要与元数据。

> 本次重构已完成：代码、主线训练流程和模型可用，优选历史版本保留在本地。按最新要求，完成条件不包含完整复现指标；已有指标只作参考。

## 重构完成与本地优选版本

按 2026-10-03 的最新要求，以代码和模型可用、保留数个本地优选版本作为结案标准。原项目与母模板保持原样，未提交或上传 Git。无需继续重训来匹配历史指标。

从完整来源目录选取六组有完整三域权重和可绑定网络定义的成熟候选，检查严格加载与有限输出；用同一批每域 512 点、种子 42 快速筛选，加热与冷却各保留两组。这里的优选限定于此次候选与抽样，不宣称穷尽全部历史版本的全局最优。筛选仅用于本地归档，不是精度复现验收。

| 保留版本 | 用途 | 本地模型包 |
|---|---|---|
| 加热带观测数据与可学习系数版本 | 保留三输出环网络和双输出肋网络，预测时同时返回系数 | [heating_with_data](docs/reference_models/selected/heating_with_data/profile.json) |
| 加热 v6.5 耦合版本 | 纯物理加热主线参考 | [heating_v65](docs/reference_models/selected/heating_v65/profile.json) |
| 冷却硬初值版本 | 硬初值冷却主线参考 | [cooling_hardconstraint](docs/reference_models/selected/cooling_hardconstraint/profile.json) |
| 冷却带观测数据与可学习系数版本 | 保留原硬初值及系数输出 | [cooling_with_data](docs/reference_models/selected/cooling_with_data/profile.json) |

包内保存三域权重、厚度权重、原网络定义、螺旋参数和 SHA-256；可获得原训练脚本的版本另存 `training_source.py`。归档预测只执行网络定义，调用时不启动原脚本的训练流程。四组模型均已成功加载并完成三域预测。

[留存索引](docs/reference_models/selected_versions.csv)、[筛选记录 R121](docs/verification/records/R121_local_selection/selection.json)、[可用性检查 R122](docs/verification/records/R122_reference_checks/verification.json) 保存在本地。`local_reference_versions/` 和 `artifacts/` 已由 `.gitignore` 忽略，模型与版本包不上传 Git。

```powershell
# 在准备好的 CPU 环境中调用；替换 bundle 与 domain 即可使用其余版本。
uv --cache-dir .cache/uv run --locked --extra cpu python tools/predict_reference.py --bundle local_reference_versions/selected/heating_with_data --domain upper --input data/processed/heating/upper_domain.csv --output artifacts/SelectedHeating/upper_prediction.csv
```

重写库的主线训练入口保持可用；追加留存版本提供原网络推理和源码参考，没有把不同版本的物理系数或训练机制合并进主线。

## 研究问题与实验设计

- 对象：外半径 0.225 m 的上下圆环，内半径 0.084 / 0.103 m；13 根螺旋肋起于 0.123 m，长 0.1534 m、宽 0.016 m、高 0.0094 m，螺旋参数 6.4105。
- 方法：冻结厚度网络提供几何及输入导数；三域网络先独立训练，再按肋 → 热流快照 → 上下环的顺序交替耦合。温变物性、对流、辐射和加热源按选定历史版本保留。
- 依据：[中面推导](docs/theory/notes/M0054.md)、[接触热阻推导](docs/theory/notes/M0053.md)、COMSOL 数据及两个历史模型包。
- 计算判据：同权重同输入预测差 ≤ 1e-3 K；固定点 PDE/BC 残差 `rtol=1e-4, atol=1e-5`；两种工况均完成短训练、保存、重载、评估、绘图，结果为有限值。
- 短预算：种子 42，上下环厚度网络各 100 步，三个温度网络各独立训练 40 步，两轮耦合每域每轮 40 步，关闭 L-BFGS。环采样 256/96/32、肋采样 128/64/32，快照 16×8×13；冷却硬初值不采样 IC 损失点。
- 评价：MSE、RMSE、MAE、最大绝对误差、Kelvin 温度的相对 L2；全点汇总按点数加权。短流程每域固定种子抽取 512 点，历史重放使用全部原评价点。

本轮不检验 H001–H005 的物理研究假设，不执行完整精度训练；计算检查通过不证明降维近似或物理公式正确。

## 作者视角（我用自己的话维护）

此段由作者维护，本次重写不代写作者理解。当前研究链、路线依据和下一项物理对照问题，留待作者复核原文与结果后记录；已确认的工程取舍见下表。

## 运行与复现

在项目根目录使用 PowerShell。Python 固定 3.12.14，独立环境 `.venv`，缓存 `.cache/uv`。依赖由 [pyproject.toml](pyproject.toml) 与 [uv.lock](uv.lock) 锁定：DeepXDE 1.15.0、PyTorch 2.12.1、NumPy 2.5.3、Pandas 3.0.6、SciPy 1.18.1、Matplotlib 3.11.2。

```powershell
uv --cache-dir .cache/uv sync --locked --extra cpu
# 每次调用保持同一 extra；辅助函数仅减少重复输入。
function prun { uv --cache-dir .cache/uv run --locked --extra cpu @args }

# 历史长表清洗；已有 True_T 不再除以厚度。
prun ring-rib-pinn preprocess --config configs/heating.toml
prun ring-rib-pinn preprocess --config configs/cooling.toml
# COMSOL 宽表：合并厚度、mm→m、积分温度→K，另存评价点。
prun ring-rib-pinn preprocess --config configs/heating_comsol.toml

# 编号必须全新；重复编号明确拒绝覆盖。
prun ring-rib-pinn train-thickness --config configs/heating_demo.toml --run-id Thickness01
prun ring-rib-pinn demo --config configs/heating_demo.toml --run-id DemoHeating01
prun ring-rib-pinn demo --config configs/cooling_demo.toml --run-id DemoCooling01

# 从指定运行整组重载，预测只需 r,theta,t；肋另需 rib_id。
prun ring-rib-pinn predict --run-dir artifacts/DemoHeating01 --domain upper --input artifacts/DemoHeating01/inputs/upper.csv --output artifacts/DemoHeating01/custom_prediction.csv
prun ring-rib-pinn evaluate --run-dir artifacts/DemoHeating01
prun ring-rib-pinn plot --run-dir artifacts/DemoHeating01

prun ring-rib-pinn legacy-evaluate --config configs/heating.toml --bundle local_reference_versions/heating_coupled --run-id LegacyHeating01
prun ring-rib-pinn legacy-evaluate --config configs/cooling.toml --bundle local_reference_versions/cooling_hardconstraint --run-id LegacyCooling01
prun python tools/verify.py --output artifacts/CheckCPU01
prun ruff check src tests tools
```

`train` 使用相同入口与配置。例如 `prun ring-rib-pinn train --config configs/heating.toml --run-id FullHeating01` 执行完整历史预算，本轮没有执行。完整配置默认使用历史厚度权重；重新训练时设置 `training.train_thickness=true` 或使用 `train-thickness`。新厚度检查点通过 `data.thickness_checkpoint` 接入，此项优先于旧权重和训练开关。

CPU/CUDA extras 互斥；CUDA 使用官方 cu130 索引：

```powershell
uv --cache-dir .cache/uv sync --locked --extra cuda
uv --cache-dir .cache/uv run --locked --extra cuda python tools/verify.py --gpu --output artifacts/CheckGPU01
uv --cache-dir .cache/uv run --locked --extra cuda ring-rib-pinn demo --config configs/cooling_demo.toml --device cuda --run-id DemoCuda01
# 切回 CPU
uv --cache-dir .cache/uv sync --locked --extra cpu
```

数据与输出路径相对 TOML 所在目录解析；CLI 的 `--run-dir/--input/--output/--bundle` 按显式命令路径解析。模块导入不读文件、不加载模型、不训练；显式运行时初始化 PyTorch 后端。

## 数据契约与模型格式

| 字段 | 含义与单位 |
|---|---|
| `r` | 半径，m |
| `theta` | 角度，rad；兼容历史 [-π,π]，网络用正弦/余弦周期化 |
| `t` | 时间，s |
| `True_T` | 厚度平均温度或明确标记的表面温度，K |
| `thickness` | 正厚度，m；厚度训练必需 |
| `rib_id` | 整数 0..12，肋数据必需 |
| `s` | 可选肋弧长，m；缺省由 r 与螺旋积分得到 |

已有 `True_T` 直接采用；显式声明 `representation="integral"` 时才将 `T[m*K]` 除以 `thickness[m]`。宽表识别 `@ t=...` 列，单位、温度表示和字段映射写在 `data.preprocess.common` 或各域覆盖项。零厚度 COMSOL 节点仅在宽表预处理时按历史规则移除；规范长表中的非正厚度报错。肋编号缺失时按 JSON 螺旋相位直接分类。

旧 COMSOL 元数据含混合编码，读取器跳过元数据字节后严格解析数值表。保留原 `cKDTree(r[m],theta[rad]), eps=0.005` 贪心抽稀，未改成真实空间距离。历史长表已经抽稀，默认不重复抽稀。`heating_comsol.toml` 写入独立目录，不替换历史评价点。

无量纲约定：`r*=r/R`、`s*=s/R`、`t*=t·alpha/R²`、`T*=(T-293.15)/900`，`alpha=15.1/(7850·460)`。旧模型包校验哈希，绑定网络定义及输入输出变换；新格式 `ring-rib-pinn/1` 保存三域与厚度网络、配置、尺度、相位和来源信息，拒绝不完整模型组、格式或预期配置不匹配。冷却硬初值包装权重不能直接交给加热 FNN。

训练目录保存配置、环境、输入哈希、评价点副本、日志、损失、阶段权重、预测、指标和图表。新运行同时保存 `code.zip` 与代码哈希。失败保留状态和 `failure.txt`；评估用运行内的同一模型组和输入，不依赖原目录。加热另报告 `surface_against_midplane`，显示域平均预测与原表面参照的差异，没有加入 MoE 表面修正；冷却无匹配表面参照，只报告域平均温度。

## 项目入口与版本来源

- [AGENTS.md](AGENTS.md)：科研协作与本项目约束。
- [hypotheses.csv](hypotheses.csv)、[experiments.csv](experiments.csv)：保留 H001–H005、R001–R010 原编号及来源内容；历史状态不等于本轮验证，新运行从 R101 起独立编号。
- [src/ring_rib_pinn](src/ring_rib_pinn)：数据、几何、物性、网络、PDE/BC、CouplingState、训练、加载、评价、绘图、CLI。
- [configs](configs)：加热/冷却完整配置、短配置、COMSOL 导入配置；[tests](tests) 和 [tools/verify.py](tools/verify.py)：计算与交付检查。
- [来源清单](docs/reference_models/source_manifest.csv)：完整外层项目 726 个文件的相对路径、大小、SHA-256 和复制位置，首轮 214 个副本、追加优选归档后 243 个副本哈希核对一致。内层 Git 重复树及缓存未重复收录。
- [模型与分支索引](docs/reference_models/index.csv)：两个主线可重放模型包与四组本地优选预测包；NTK、MoE、soft-MoE、可学习物理系数和表面修正保留原始来源，不追加重写训练机制。
- `local_reference_versions/source/`：原始代码、推导和日志；历史 CSV 中的证据路径相对完整来源根目录，在此副本下查找。未复制的额外图像、数据和权重按来源清单定位原存档。
- `data/raw/`、`data/processed/`、`artifacts/<run_id>/`：本地数据与结果，默认忽略。原项目及母模板保持原样。

| 工况/阶段 | 保留来源 |
|---|---|
| 加热独立环 | `Re-PINN/pinn_no_rib_data_v6.5.py` |
| 加热 Ring → Rib | `Re-PINN/pinn_ring2rib_v6.py` |
| 加热耦合 | `Re-PINN/pinn_v6.5.py`、`models_heating/` |
| 冷却三个阶段 | 对应 `v6_colding_hardcore.py`；耦合权重来自 `models_colding_linux/` |
| 厚度 | `ThicknesNet/data_thickness_train.py`、独立厚度 CSV、`Re-PINN/ring_thickness/` |

## 当前验证结果

2026-10-03 的 [R119 CPU 检查](docs/verification/records/R119_cpu_checks/verification.json) 与 [测试日志](docs/verification/records/R119_cpu_checks/tests.md)：30 项通过，覆盖预测、独立/耦合 PDE/BC、快照、无量纲往返、周期、弧长、冻结厚度导数、初值、重载及失败场景；固定点包含源项区域、边界端点、辐射分区与接触中心。历史重放另检查原输入不可用后仍可重载评估。旧参考函数有一次 searchsorted 非连续张量性能提示，不影响数值判据。

| 运行 | 点集 | 上环 RMSE/K | 下环 RMSE/K | 肋 RMSE/K | 全点 RMSE/K |
|---|---|---:|---:|---:|---:|
| [R110 加热历史重放](docs/verification/records/R110_heating_legacy/metrics.json) | 原 397239 个域点 | 2.776 | 2.279 | 1.406 | 2.493 |
| [R111 冷却历史重放](docs/verification/records/R111_cooling_legacy/metrics.json) | 原 77217 个域点 | 2.206 | 3.804 | 5.904 | 3.691 |
| [R108 加热短训练](docs/verification/records/R108_heating_smoke/metrics.json) | 每域固定 512 点 | 540.324 | 74.710 | 302.896 | 360.221 |
| [R109 冷却短训练](docs/verification/records/R109_cooling_smoke/metrics.json) | 每域固定 512 点 | 19.542 | 48.317 | 38.519 | 37.417 |

[R116 加热](docs/verification/records/R116_heating_legacy_portable/metrics.json) 与 [R117 冷却](docs/verification/records/R117_cooling_legacy_portable/metrics.json) 保存了同一历史重放的可搬移输入和完整检查点，指标与 R110/R111 一致，可以直接使用 `evaluate/predict --run-dir`。

[R118 便携性检查](docs/verification/records/R118_delivery_checks/verification.json) 将两种短训练运行搬移，并从其他启动目录执行预测、评估、绘图；重载与 CSV 读回的最大预测差约 1.14e-13 K，阶段 2 与最终模型参数完全一致。此前 R115 暴露 float32 CSV 简写精度损失，已通过导出前无损转为 float64 修复，失败记录保留。数值网络和 Kelvin 转换没有改变。

[R120 独立厚度训练](docs/verification/records/R120_thickness_smoke/status.json) 已完成上下环各 100 步并保存 `thickness.pt`。真实 COMSOL 宽表导入结果见 [预处理记录](docs/verification/records/data/comsol_heating/preprocess.json)，上/下环/肋分别为 31395 / 34230 / 15078 行。

历史指标由迁入数据与权重重新计算；短训练是不同预算、不同点集的流程检查，不能当作公平精度对照。加热历史表面参照的上/下环 RMSE 为 5.386 / 6.135 K，单独保存在 R110 表面指标。

短流程训练分别约 30.62 / 30.99 秒，只是本机执行记录。R101 因表面路径错误失败、R104 因元数据编码失败、R105/R107 因短厚度续训出现非正值失败，全部保留；R102 是较早的温度点集厚度预训练检查。[R113 GPU 核验](docs/verification/records/R113_gpu_checks/verification.json) 已在 RTX 5060 Laptop、PyTorch 2.12.1+cu130 上通过两组模型的加载、PDE 前后向与精确保存重载；环境已恢复 CPU，CUDA 包保留在项目缓存。本轮未执行 GPU 完整训练。

## 已采纳的关键取舍与物理疑点

| 编号 | 日期 | 决定与依据 | 状态 |
|---|---|---|---|
| D09 | 2026-10-03 | 按用户计划建立独立参考库，采用“03-科研项目”、TOML、uv；原库与模板只读 | 已实施 |
| D10 | 2026-10-03 | 保留加热/冷却 Lambda、独立/耦合肋 BC 及温度钳制的版本差异 | 已实施并对照 |
| D11 | 2026-10-03 | 验收限于流程和历史重放，完整预算本轮不执行 | 用户计划已确认 |
| D12 | 2026-10-03 | 短厚度续训使用历史权重、全量独立厚度点、1e-5 学习率；原 1e-3 短预算出现非正值，失败保留。完整配置仍为 1e-3 | 调整流程超参，公式与步数不变 |
| D13 | 2026-10-03 | 按最新用户要求，以代码与模型可用、保留本地优选版本作为重构完成标准；不要求完整复现指标，不上传 Git | 已完成，见 R121/R122 |

以下问题按原代码保留，修正应另建配置和对照：

1. 厚度对 `r*` 的导数用于斜率时缺少一项物理尺度换算，与角向导数单位处理不同。
2. 高斯接触掩码峰值约 1.596，`1-mask` 可为负，未截断到 [0,1]。
3. 肋扩散为 `kappa*T_ss`，未改为完整变导热率通量散度；热阻中 `1e-6` 保留为代码常数，不能直接等同于任意界面热阻参数。
4. 肋编号训练域是连续矩形，几何映射时转整数，网络特征仍使用连续编号，未改成离散采样。
5. 快照查询实际为 `searchsorted` 上节点，保留该行为；各次运行用独立状态对象。
6. 冷却初值内保留原 793.15、0.225、0.084 常数，改几何时须检查适用性。主线不含 NTK、MoE、可学习物理系数与表面恢复修正。

## 一轮工作、隐私与发布

先确定问题、比较对象、配置、指标及预算；用新 run_id 执行并保存负结果；计算与研究结论分别核对，由负责人决定是否更新假设。作者视角由作者回写。

大型数据、权重、原始推导和本地产物默认忽略，索引不代表再发布许可。本轮未提交、推送、安装 Hook 或修改 Git 设置；发布前另核查完整拟发布内容、署名与数据权限。

给 Agent 的启动语：请读 README、AGENTS 和相关 CSV 行，回查本轮版本的原文、配置和结果。按已确定预算执行有编号的实验，保存失败和变化原因；报告证据及边界，不将执行完成写成假设成立。



## 研究材料整理

[模型、现有推导与缺口](docs/theory/README.md)汇合代码目录、毕业设计、历史备份和文献中的相关材料，保留仓库内整理稿、来源编号与哈希。本轮只整理既有内容并列出缺口，未补写证明、重新训练或调整假设支持状态。详见本项目资料入口。

## 开源许可与集合布局

[English](README.md) · [MIT](LICENSE) · [贡献说明](CONTRIBUTING.zh-CN.md)

公开范围内的项目自有代码与原创整理稿采用 MIT；依赖保留原许可，历史数据、模型与第三方原件不纳入本次发布授权。迁移后的 uv 环境按原锁文件重建，科学假设状态不变。
移动后实际检查：[MOVE20261004_01](docs/publication/relocation.json)。
