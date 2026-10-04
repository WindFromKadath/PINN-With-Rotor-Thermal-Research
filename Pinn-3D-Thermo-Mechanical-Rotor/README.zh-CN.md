# Pinn-3D-Thermo-Mechanical-Rotor

本项目归入 Rotor-Thermal-Research 集合，按[发布范围](docs/publication/README.zh-CN.md)准备提交；数据、权重与原件保持忽略。

新克隆请先读[安装与检查说明](docs/running_from_clone.md)；[必要理论文本](docs/theory/README.md)已随仓库保存。历史重放与真实数据示例需要另行恢复本地数据、网格及权重，下方验收和模型链接指向仓库内摘要与元数据。

[English](README.md) | 简体中文

可独立使用的转子 PINN 科研参考库：3D 瞬态热学、热→力单向耦合、轴对称热弹性、纯数据 MLP、IWG 刚体处理，以及稳态三域 X-PINN。

2026-10-04 已完成代码与模型迁移。首轮验收针对软件、历史模型和短流程可用性；没有执行完整预算精度复现。原目录与母模板保持原样，全部数据、权重和运行产物仅在本地保存。

## 研究范围与来源

采用科研项目模板的双语入口、维护记录、协作规则、假设与实验索引。完整来源包括 pinn-3d-thermo-mechanical-rotor 源码快照，以及同级 LittleTinyTest、FXXKINGGOD 的数据与权重。

热网输入 `[r/L,z/L,t/tau,sin(theta),cos(theta)]`，输出 `(T-T0)/dT`；力学网输入 `[r/L,z/L,t/tau]`，输出 `[u_r/U,u_z/U]`，其中 `U=alpha*dT*L`。力学采用随温度变化的静力平衡，没有位移惯性项和周向位移输出。温度桥冻结参数，保留空间和时间输入导数；默认查询 theta=0。

主线与正式热学的 `ring.stl` 实际覆盖下环：z 约为 -34.396 至 -1.396 mm。代码中的较大几何包围盒不代表已在完整转子上训练。X-PINN 使用下环、肋和上环各自的 STL，属于稳态温度路线。

| profile | q / W·m⁻² | h / W·m⁻²·K⁻¹ | dT / K | tau / s |
|---|---:|---:|---:|---:|
| little_transient、fxx_transient | 100000 | 20 | 300 | 200 |
| formal_thermal、xpinn_steady | 50000 | 50 | 100 | 100 |
| iwg | 使用 400→300 K 径向温度场 | 不适用 | 106.85 | 200 |

各模型包绑定网络、尺度、输出变换和参考系，不按相同文件名混用权重。NTK/ReLoBRaLo、MoE、SDF 和其他探索代码保存在来源快照与索引中，本轮不重写其完整训练对照。

## 安装与运行

Python 固定 **3.13.14**，依赖由 uv.lock 锁定。CPU 与 CUDA extras 互斥，分别使用 PyTorch 官方 CPU/cu130 索引。缓存位于项目 `.cache/uv`。

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location).Path '.cache/uv'
uv sync --locked --extra cpu
uv run --locked --extra cpu thermo-rotor --help
uv run --locked --extra cpu thermo-rotor demo --config configs/demos/coupled.toml
uv run --locked --extra cpu pytest -q
```

先进入项目目录执行这些命令。CLI 默认配置从安装项目解析，配置中的路径相对配置文件解析；也支持从其他目录使用 `uv run --project <项目目录> --locked --extra cpu ...`。

| 功能 | `uv run --locked --extra cpu` 后的命令 |
|---|---|
| 热→力完整短流程 | `thermo-rotor demo --config configs/demos/coupled.toml` |
| 单独热学 | `thermo-rotor train-thermal --config configs/demos/thermal.toml` |
| 单独力学 | `thermo-rotor train-mechanical --config configs/demos/mechanical.toml` |
| MLP 两组缩小标签预算 | `thermo-rotor train-mlp --config configs/demos/mlp.toml` |
| IWG 约束与投影 | `thermo-rotor train-iwg --config configs/demos/iwg.toml` |
| 稳态 X-PINN Stage A–D | `thermo-rotor train-xpinn --config configs/demos/xpinn_steady.toml` |
| 正式热学独立工况 | `thermo-rotor demo --config configs/demos/formal_thermal.toml` |
| COMSOL 宽表转长表 | `thermo-rotor preprocess --config configs/preprocess.toml` |
| STL 体内与边界采样 | `thermo-rotor sample --config configs/demos/coupled.toml` |
| 指定运行重载评价 | `thermo-rotor evaluate --run-dir artifacts/R102` |
| 指定运行重建图 | `thermo-rotor plot --run-dir artifacts/R102` |
| 指定运行预测 | `thermo-rotor predict --run-dir artifacts/R102 --model thermal --input artifacts/R102/thermal_inputs.csv --output artifacts/new_prediction.csv` |
| 历史主线重放 | `thermo-rotor legacy-evaluate --bundle little` |
| 运行指标汇总 | `thermo-rotor compare --run-dirs artifacts/R102 artifacts/R108 --output artifacts/new_comparison.json` |

未指定 run_id 时自动生成新编号；重复编号与已有预测/转换输出拒绝覆盖。失败保留目录、配置和 failure.txt。单独力学示例默认绑定 `artifacts/R102/thermal.pt`；使用自己的热学运行时修改 `paths.thermal_checkpoint`。

CUDA 小检查采用独立 `.venv-cuda`，保持 CPU 环境完整。需要新的 GPU 运行时可在单独 PowerShell 进程中设置 `$env:UV_PROJECT_ENVIRONMENT='.venv-cuda'`，再执行 `uv sync --locked --extra cuda --no-dev` 和 `uv run --locked --extra cuda --no-dev thermo-rotor demo --config configs/demos/coupled.toml --device cuda`。

`configs/demos/` 使用 seed=42：热学和力学各 Adam 40 步，域内 128 点、每边界 64 点、IC 64 点，3 个评价/锚点时间片，每片至多 32 个训练锚点。MLP 为 32/96 个总标签；原 1100/33000 版本均为约 66,818 参数网络。IWG 每阶段 20 步，X-PINN 每阶段 20 步；短流程不使用 L-BFGS。另有测试覆盖小规模固定闭包 L-BFGS。

`configs/` 完整配置保留主要来源预算，首轮未执行；原 YAML 一并归档。新参考训练使用显式固定的小批次和日志，源采样/优化器的完整精度复现未验收。IWG 短流程第三阶段采用 Adam 检查完整损失，完整配置第三阶段采用 L-BFGS；比较预算时须包括这个差别。

## 数据契约与模型接口

内部 SI 字段：x/y/z/r 为 m，theta 为 rad，t 为 s，T 为 K，u_r/u_z 为 m，stress 为 Pa。导入器支持 COMSOL `表达式 (单位) @ t=...` 宽表及整理长表；明确声明长度、位移、温度、时间单位。负时间、非正半径/温度、非有限值、缺列、单位不符及不完整位移对均报错。T_true 映射为 T，不改变已是 Kelvin 的温度。

CSV 分块读取，锚点采用按时间独立的有界随机优先级抽样，记录实际点集；这是新参考库的抽样协议，不宣称与旧 reservoir 算法选中了相同空间行。训练与评价使用不同种子，但历史训练/测试划分有不确定性，不能仅凭不同种子宣称严格无泄漏。

默认 preprocess 配置使用真实 COMSOL 表的前 32 个空间行验证导入。全量转换另用 `configs/preprocess_full.toml`，可能生成数 GB 长表，本轮不执行。

必要原表、STL 和模型已迁入。约 5 GB 的 ElsMisesSCheck.csv 仅建立来源哈希索引；当前没有完整应力误差重算。热学温度和位移指标来自指定评价快照；正式热学和稳态 X-PINN 暂无匹配 COMSOL 参照，输出图和 T0 诊断指标只用于流程检查。

```python
from thermo_rotor.models.checkpoint import load
from thermo_rotor.evaluation.predict import predict

model, profile, metadata = load("artifacts/R102/thermal.pt", expected_kind="thermal")
# frame 使用 SI r,z,t,theta 字段。
values_K = predict(model, frame, profile, metadata["spec"], metadata["gauge"])
```

导入不读取数据、不加载模型、不创建目录、不启动训练。检查点由 `.pt` 张量文件与同名 `.json` 定义共同组成；搬移时连同定义与评价输入保存。加载验证格式、严格网络结构、有限权重、SHA-256 和调用方指定的工况。评价只读取指定运行中的模型与输入快照，不依赖旧来源路径。

## 历史参考版本与验证证据

本地包索引见 [index.json](docs/reference_models/index.json)，来源文件与哈希见 [source_manifest.csv](docs/reference_models/source_manifest.csv)。保留 8 个包、10 个模型，原 `.pth` 与可独立加载的 `.pt/.json` 同时保存。

| 包 | 用途与选择依据 |
|---|---|
| little | 首选主线热学＋力学；同权重新旧预测差为零 |
| fxx | 同来源匹配热学＋力学，保留主线对照 |
| mlp_1100、mlp_33k | 原标签量对照，诊断样本优先使用 mlp_33k |
| iwg_fxx_constraint | 首选约束候选，恢复来源显式参考点，有限样本优于 Little 原评价器 |
| iwg_constraint、iwg_projection | 保留 Little 原始评价语义及投影对照 |
| fxx_ntk | 已加载的历史 NTK 对照，有限诊断较差，明确不作为优选模型 |

优选依据限于匹配工况、来源 best 权重和固定小样本，不宣称整个来源文件夹的全局最优。IWG constraint 的 Little 原评价器没有恢复训练参考点，原样重放因此可能出现较大误差；FXX 候选显式恢复约束点。投影使用记录在检查点中的单一全局轴向平移量，后续分批预测保持一致；这不构成物理修正或线性拟合精度证据。

- [R101 历史兼容检查](docs/verification/records/R101/verification.json)：9 个初选模型有限预测，温度差为零、位移最大差约 1.1e-11 m，主线 PDE 对照通过；[R120](docs/verification/records/R120/metrics.json) 补充第 10 个优选约束模型。
- R102–R106：热→力、MLP、IWG、稳态 X-PINN 和正式热学均完成短训练、保存重载、评价和绘图。短训练数值有限；IWG 短预算损失仍很大且波动，不据此认定收敛。
- R105 首次移植遗漏原版 380 K 温度锚点；已补回并登记 R125，后者是稳态 X-PINN 的交付参考，旧结果保留。
- [R107 交付检查](docs/verification/records/R107/verification.json)：774 个迁入副本哈希一致，其他启动目录 CLI、可搬移运行和覆盖保护通过；8 个历史包重新加载并生成本地结果。
- R102/R108 同种子 CPU 短流程的损失差为零，预测 CSV 哈希相同。实现测试验证导数、解析场、冻结桥、界面停止梯度、单位、几何、重载和失败场景。
- 最终 CPU/GPU 状态与环境见 [总验收记录](docs/verification/records/migration_20261004/verification.json)，运行登记见 [experiments.csv](experiments.csv)。

## 局限、维护与权限

无惯性力学、theta=0 温度桥、来源常物性、法向翻转和源版轴向伸长目标等假设均保留；轴向伸长公式的尺度疑点留待独立实验。X-PINN 的源侧通量先求导再 detach，Stage D 仍保持 lower→rib→upper 的界面梯度方向，没有改为对称双向界面损失。

迁移完成表示代码、模型与参考流程可用，不验证 PINN 完整精度、收敛性、材料参数适用性或替代 COMSOL 的能力。假设科学状态由负责人判断，软件检查完成不会自动标为 supported。

维护与已采纳取舍见 [MAINTAINER.md](MAINTAINER.md)、[AGENTS.md](AGENTS.md) 与 [贡献指南](CONTRIBUTING.zh-CN.md)。作者理解由作者维护。来源资料、数据和权重没有已核实的公开再发布许可；代码与必要说明按本项目提交范围准备，默认忽略大文件和私有快照，没有提交、推送或修改 Git 配置。


## 研究材料整理

[现有推导与研究缺口](docs/theory/README.md)按模型、尺度、版本和证据整理本地来源。本轮未补写证明或重新训练，科研假设状态未改动。必要整理稿、来源哈希与验收摘要已纳入仓库；原件及详细归档继续本地保存。

## 许可

公开范围内的项目自有代码与原创整理稿采用 [MIT](LICENSE)。安装的依赖保留自己的许可；忽略的数据、权重与第三方原件不在本次发布范围内。
移动后实际检查：[MOVE20261004_01](docs/publication/relocation.json)。
