# Pinn-Inverse-Rotor-Thickness · 科研参考库

本项目归入 Rotor-Thermal-Research 集合，按[发布范围](docs/publication/README.zh-CN.md)准备提交；数据、权重与原件保持忽略。

新克隆请先读[安装与检查说明](docs/running_from_clone.md)；[必要理论文本](docs/theory/README.md)已随仓库保存。历史重放与真实数据示例需要另行恢复本地数据、网格及权重，下方验收和模型链接指向仓库内摘要与元数据。

已知厚度先验下的上下圆环温度重建与 13 条全长肋接触耦合。首版是可运行、可检查、可复用的二维圆环—一维肋 PINN 实现；未知厚度反演留待后续研究。

当前验收按用户 2026-10-03 的补充要求：保证参考代码、完整流程和已训练模型可用，保留少量有比较依据的历史版本，全部本地保存。精确重现旧报告指标不再是完成条件。新库的训练/评估/绘图流程已通过；历史模型按各自定义独立推理，不直接替换新模型格式。

本地保留 3 组模型，版本定义见仓库内[模型索引](docs/reference_models/index.csv)，优先使用 `fin_v3_1`。每组包含来源代码、独立网络工厂、3 个温度权重、2 个物理厚度权重、输入尺度、哈希和可运行示例；旧目录保持原样。

## 研究问题与实验设计

- 问题：给定空间厚度先验及温度观测，能否在未见空间位置重建温度？全长耦合是否改善空间外推？
- 范围：PyTorch + DeepXDE；两个厚度网络、上下圆环两个温度网络、覆盖整数编号 `0..12` 的一个共享肋温度网络。
- 验证：解析函数检查时空导数、厚度梯度与守恒残差；独立积分检查弧长；匹配求积检查交换功率；合成 demo 检查训练、保存、重载、评估及绘图。
- 停止条件：配置中的固定训练预算；遇到非有限值终止并保留失败目录。短示例完成表示流程可用，不表示 PINN 或耦合已收敛。
- 真实实验：恢复 COMSOL 数据后，先登记独立域基线、预算与精度门槛，再比较空间测试集。`H001`、`H002` 保持 `untested`，计算检查不替代真实研究证据。

沿用科研项目模板（来源名称保留，阅读内容已纳入本库）的 README、AGENTS、假设和实验索引结构。`completed` 只表示执行完成，支持程度在假设表中另行判断。

## 作者视角（我用自己的话维护）

- 当前研究链：待作者补充问题、假设、对照、证据与局限。
- 当前为什么走这条路线：待作者维护；已采纳决策见下文。
- 下一步要弄清：待作者补充自己的预测和复核计划。

作者理解后自己回写；AI 的实现说明不替代作者判断。

## 运行与复现

在 PowerShell 中进入项目根目录。Python 固定 **3.12.14**，使用独立 `.venv`；`pyproject.toml` 声明依赖，`uv.lock` 锁定版本及 PyTorch 官方索引。

```powershell
Set-Location Pinn-Inverse-Rotor-Thickness  # 从总仓库根目录进入
$env:UV_CACHE_DIR = Join-Path (Get-Location).Path '.cache/uv'
uv sync --locked --extra cpu
uv run --locked --extra cpu rotor-pinn demo --config configs/demo.toml
uv run --locked --extra cpu pytest
uv run --locked --extra cpu ruff check .
```

项目也配置了本地 uv 缓存，但现有环境变量可能覆盖它。上述进程级设置确保本机使用 `.cache/uv`；可用 `uv --cache-dir .cache/uv ...` 替代。不改全局 uv 配置。

GPU 使用互斥的 `cuda` extra，官方 **cu130** 索引：

```powershell
uv sync --locked --extra cuda
uv run --locked --extra cuda python tools/verify.py --cuda --run-id V_NEW_CUDA
uv run --locked --extra cuda rotor-pinn demo --config configs/demo.toml --device cuda
```

同一个 `.venv` 每次选择一个 extra，切换时替换 PyTorch 构建，后续命令保持相同 extra。禁止同时选择 `cpu` 与 `cuda`。uv 官方方案（定义与操作已写入本页），PyTorch 2.12 构建说明（定义与操作已写入本页）。

| 用途 | 根目录命令（示例为 CPU） |
|---|---|
| 转换 COMSOL/历史长表 | `uv run --locked --extra cpu rotor-pinn preprocess --config configs/reference.toml` |
| 仅厚度预训练 | `uv run --locked --extra cpu rotor-pinn train-thickness --config configs/reference.toml` |
| 三域训练与耦合 | `uv run --locked --extra cpu rotor-pinn train --config configs/reference.toml` |
| 无原数据完整示例 | `uv run --locked --extra cpu rotor-pinn demo --config configs/demo.toml` |
| 重载评估全部分区 | `uv run --locked --extra cpu rotor-pinn evaluate --run-dir artifacts/R002_CPU_DEMO --split all` |
| 重建图表 | `uv run --locked --extra cpu rotor-pinn plot --run-dir artifacts/R002_CPU_DEMO` |
| 保存解析检查 | `uv run --locked --extra cpu python tools/verify.py --run-id V_NEW_NUMERICS` |

自动生成独立 `run_id`，也可指定 `--run-id MY_RUN`；重复编号拒绝覆盖。评估默认 `--split test`，读取运行目录配置、权重和输入快照，无需原数据路径仍然存在。TOML 中路径相对**配置文件目录**解析。

`demo.toml`：120 步厚度预训练、每个温度网络 40 步独立训练、两轮各 40 步耦合，L-BFGS 关闭。`reference.toml`：2000 步厚度、2000 步独立温度、5 轮各 1500 步耦合、最多 5000 次 L-BFGS；真实精度待验证。研究可改 `dtype="float64"`；L-BFGS 使用固定配点与完整训练观测以保持线搜索确定性。

## 数据契约与接入

| 数据 | 必需字段 | 单位/约束 |
|---|---|---|
| 上/下圆环温度 | `r,theta,t,True_T` | m、rad、s、K；温度正，时间非负 |
| 厚度 | `r,theta,thickness` | m、rad、m；厚度正 |
| 肋温度 | `r,theta,t,True_T,rib_id` | 同上；编号是配置范围内整数 |

圆环温度表可附加 `thickness`。若无厚度列，配置 `[data] upper_thickness_csv/lower_thickness_csv` 指定独立厚度表，或 `thickness_checkpoint` 指定 `train-thickness` 产物 `thickness.pt`。厚度单独预训练无需温度数据。厚度检查点的几何、网络及输出尺度须匹配配置。

`preprocess` 支持 COMSOL 宽表（如 `T (K) @t=120`）和历史长表。显式指定 m/mm、K/C；角度须为 rad，时间须为 s。x/y 可转换为 r/theta。列映射为**规范字段 = 原列名**：

```toml
[preprocess]
length_unit = "mm"
thickness_unit = "mm"
temperature_unit = "C"
encoding = "utf-8-sig"
skip_rows = 0

[preprocess.columns]
x = "x (mm)"
y = "y (mm)"
thickness = "H (mm)"
# 长表可另设 True_T = "T"，t = "Time"
```

按实际导出头设置映射和 `skip_rows`，不猜测单位。输入为 `preprocess.upper_input/lower_input/rib_input`，输出为 `data.upper_csv/lower_csv/rib_csv`。先验证三表再写新文件；禁止覆盖任何输入；已有输出需显式 `--overwrite`。缺列、空表、非有限值、非正厚度、非法半径/肋编号立即报错。

角度按 $2\pi$ 归一化；肋表没有编号时按解析曲线的圆周角距离分配最近肋，恢复真实数据时应核查映射。按 `(r,theta[,rib_id])` 空间位置分组，坐标取 10 位小数、种子 42，默认 70%/15%/15%；同一位置全部时间记录同区。分组数量取整使比例近似，至少需要三个空间组。预训练与温度观测仅使用 train，验证/测试指标另行计算。DeepXDE 控制台的 `Test loss` 是配点损失，不是空间测试集误差。

demo 在运行目录生成两圆环各 128 个空间位置、13 条肋各 10 个位置、9 个时刻（0–1200 s）的合成数据。标签为平滑函数，**没有假设满足本 PDE**，因此其 RMSE 只描述短流程拟合，不验证求解精度。历史权重及温度表已在同级 `PINN_Inverse_fin_pls` 目录找到，见下文历史模型复用；新全长接触实现尚未使用这些数据完成研究训练。

## 模型、几何与适用范围

解析肋曲线：$\theta_i(r)=1.2209+6.4105(r-0.123)+2\pi i/13$，$r\in[0.123,0.225]$ m。

\[
s(r)=\int_{0.123}^{r}\sqrt{1+(6.4105\xi)^2}\,d\xi,
\qquad L=0.15334921214568\ \mathrm m.
\]

上/下圆环内半径 0.084/0.103 m，外半径 0.225 m；肋宽 $w=0.016$ m、高 $h_b=0.0094$ m，截面积 $A=wh_b$ 恒定。历史拟合 CSV 不自动覆盖此参考几何。

圆环能量残差（物理坐标，W/m²）：

\[
\rho c(T_j)H_j\partial_tT_j
-\frac1r\partial_r(r k(T_j)H_j\partial_rT_j)
-\frac1{r^2}\partial_\theta(k(T_j)H_j\partial_\theta T_j)
-H_j Q_j+f_j\ell(T_j)+S_j=0.
\]

肋残差（W/m）：

\[
\rho c(T_b)A\partial_tT_b
-\partial_s(k(T_b)A\partial_sT_b)-A Q_b
+P_{\mathrm{exposed}}\ell(T_b)-q'_u-q'_l=0.
\]

$\ell(T)=h_c(T-T_\infty)+\varepsilon\sigma(T^4-T_\infty^4)$。每个圆环一个端面暴露，$f_j=1$；肋两个侧面暴露，$P_{exposed}=2h_b$；肋两端绝热。圆环内外径边缘为对流/辐射 Robin 边界。假设保存在 `physics.exposed_ring_faces/rib_exposed_sides/rib_end_condition/ring_radial_boundary`；不额外计算接触面的环境散热。

默认 $T_\infty=293.15$ K，$h_c=10$ W/(m² K)，$\varepsilon=0.3$，密度 7850 kg/m³。上环体热源 $Q_u=Q_0e^{-(r-r_{in,u})/d}$，限于内缘三倍衰减长度以内；下环使用自身内缘和倍率 0.3；肋体热源零。研究设置 $Q_0=10^8$ W/m³、$d=0.005$ m；demo 较小热源。k、c 温度节点见 `config.Physics`，区间内线性插值，外部截断到端值。完整默认值保存在运行快照；真实研究需确认物性和工况。

网络坐标：圆环 `[r/R,theta,t/t_max]`，肋 `[s/R,rib_id,t/t_max]`。肋只采整数编号，时间在第三列。内部角度使用 sin/cos 特征，保证圆环周期一致；时间使用平滑变换。$T=T_\infty+T_{scale}u$，输出变换强制 $u(0)=0$，因此仅支持初始均匀环境温度。

厚度为 `minimum_thickness + scale * softplus(network)`，输出 m，尺度随权重保存。冻结参数保留对输入坐标的求导，扩散残差保留 H 和 k 的空间梯度。厚度先验静态，同一位置重复时刻不应给矛盾厚度；首版不从温度反演 H。

## 全长接触与守恒

两侧默认 $R_{contact,j}=0$，平均温度之间仍有半厚度降维热阻：

\[
R_{eff,j}=\frac{\overline H_j}{2k(\overline T_j)}+\frac{h_b}{2k(T_b)}+R_{contact,j},
\qquad q'_j=\frac{w}{R_{eff,j}}(\overline T_j-T_b).
\]

$R_{eff}$ 单位 m² K/W，$q'$ 单位 W/m；正值表示圆环向肋传热。温差/热阻关系参照 COMSOL 接触换热定义（定义与操作已写入本页），半厚度串联阻力和高斯平均是本参考模型的降维假设。零界面热阻不强制降维域平均温度相等。

肋全长用 Gauss–Legendre 弧长节点，圆环用径向 Gauss–Legendre、等间隔角度及物理面积权重 $\omega_i=r\,dr\,d\theta$。核 $K_{ia}$ 由 $\exp(-|x_i-x_a|^2/\sigma_G^2)$ 除以同一面积求积的积分归一化；同一核用于圆环平均与投影：

\[
\overline T_{j,a}=\sum_i\omega_iK_{j,ia}T_{j,i},
\quad S_{j,i}=\sum_aK_{j,ia}\lambda_aq'_{j,a},
\quad\sum_i\omega_iS_{j,i}=\sum_a\lambda_aq'_{j,a}.
\]

$\lambda_a$ 为弧长积分权重，求和含全部肋；H 使用同样平均。圆环物理源 $-S_j$，肋物理源 $q'_u+q'_l$。诊断以 $\sum_a\lambda_a|q'_a|$ 归一化以防正负抵消。守恒是同一离散求积网格上的关系，不保证连续积分、PDE 损失或三维真解同样精确；真实研究应进行核宽、面积和弧长网格加密检查。

显式 `CouplingState` 保存时间网格和两侧线功率。先独立训练温度，按当前三域温度更新全长接触功率，默认松弛 $q'_{new}=0.5q'_{calculated}+0.5q'_{old}$，再交替训练三域。每轮使用同一份冻结功率；时间/弧长查询线性插值，肋端超出 Gauss 节点时取端值。最终状态为最后一轮训练用的滞后功率，有限预算和离散功率守恒不能证明耦合固定点收敛。

该模型忽略厚度方向温度细节、真实三维接触区域与局部接触压力，仅适用于降维、边界和初温假设合理的场景。

## 产物与评价

每次保留配置、manifest、环境、控制台日志、输入快照、损失、厚度/温度权重、`coupling.npz`、守恒/重载诊断、三域预测、指标及 `figures/`。独立厚度运行仅生成预训练产物。manifest 记录包源码 SHA256、数据快照/原输入 SHA256、配置/权重/uv.lock 哈希和种子；失败保留 `failure.txt`。配置与权重内嵌配置不一致、权重/输入哈希不符时拒绝评估。

四指标：RMSE、MAE、最大绝对误差单位 K；$L2=\|\widehat T_K-T_K\|_2/\|T_K\|_2$ 的分母使用绝对 Kelvin 温度。不能与温升分母混用。总体按所有样本聚合；各域及 train/validation/test 分别报告。损失记录厚度采样步与 DeepXDE 展示步；各域残差先按物理尺度归一化，观测再加权，原始 W/m² 和 W/m 不能直接比较。

验收产物见 [experiments.csv](experiments.csv)。`tools/verify.py` 检查 float64 相对误差小于 $10^{-6}$；pytest 覆盖数据、几何、导数、功率、模型重载、L-BFGS 小流程和错误处理。真实 COMSOL 精度与基线比较仍待验证。

2026-10-03 的实际验收：

| 记录 | 结果与范围 |
|---|---|
| `R002_CPU_DEMO` | 厚度与三域训练、全长交换、保存/重载、三分区评估和图表均完成；重载预测最大绝对差为 0 K |
| `R003_SI_REUSE` | 合成快照经过 preprocess 生成 SI CSV，以 train 入口复用已保存厚度权重并完成评估/绘图；验证真实数据接入路径的接口，不涉及真实 COMSOL |
| `V003_NUMERICS_FINAL` | float64 圆环解析残差相对误差约 1.49e-16；厚度导数误差为 0；随机正负线功率离散不平衡不超过 2.8e-17；附最终 CPU pytest 报告（28 通过、1 CUDA 跳过） |
| `V002_CUDA` | Python 3.12.14 / torch 2.12.1+cu130 / RTX 5060 Laptop；矩阵计算有限，解析自动微分误差为 0，GPU 圆环解析残差通过；附 CUDA 单项及完整 29 项通过报告 |
| `LBFGS_PIPELINE` | 首轮小流程因 DeepXDE BC/test 缓存不匹配而失败；原失败与报告保留，修正后的 L-BFGS 小流程已通过 pytest |

`R002` 的合成空间测试集总体 RMSE 约 **8.7365 K**（540 个样本），仅表示短预算下对非 PDE 真解标签的拟合结果。当前 `.venv` 已恢复为 CPU 构建；GPU 验证证据保留，运行 GPU 前按上文切换 extra。数值/执行验收完成没有改变 `H001/H002` 的待验证状态。

## 项目入口

| 位置 | 内容 |
|---|---|
| `src/rotor_pinn/config.py` | TOML / dataclass / 相对路径 |
| `data.py / geometry.py` | 数据适配、分组、弧长与求积 |
| `materials.py / networks.py / physics.py` | 物性、共享网络工厂、守恒自动微分 |
| `coupling.py / training.py` | 显式全长接触与交替训练 |
| `evaluation.py / plotting.py` | 重载、分区指标、非交互图表 |
| `artifacts.py / backend.py / cli.py / synthetic.py` | 记录、设备、命令与合成算例 |
| `configs/ / tests/ / tools/verify.py` | 示例/研究配置、验收及持久化检查 |
| `data/raw/ / data/processed/` | 私有原数据和处理数据 |
| `artifacts/<run_id>/` | 本次配置、权重和原始结果 |
| `src/rotor_pinn/legacy.py` | 历史模型独立加载、SI 预测和自动微分接口 |
| `tools/curate_legacy.py / reuse_preserved.py / verify_preserved.py` | 本地筛选、CSV 推理和独立移植验证 |
| `local_reference_versions/` | 筛选后的 3 组来源代码、权重与示例；整个目录忽略 Git |
| [AGENTS.md](AGENTS.md) | 协作与隐私规则 |
| [hypotheses.csv](hypotheses.csv) | 假设、预测、范围及支持程度 |
| [experiments.csv](experiments.csv) | 实际运行及验证证据 |

导入模块不读取数据、训练或绘图；DeepXDE 仅在显式初始化/网络构造时载入。沿用模板原 CSV 列，不另建平行研究日志。

## 历史来源与已采纳取舍

只读参照原项目（来源名称保留，阅读内容已纳入本库），参考提交 `77b4754`，包含 `InversePINN_Again`、`PINN_Inverse_fin_pls` 的厚度预训练、圆环和肋模型。新库统一 SI 厚度、输入顺序、整数编号和保存/加载工厂，以接触温差定义交换而非用纵向肋热流充当接触源，没有复制整套历史版本。

历史报告总体 RMSE **6.0609 K** 仅作原模型与数据的背景，不是新实现验收承诺。新示例数据、预算及划分不同，不能直接比较。

### 已恢复的历史模型与只读复用

2026-10-03 扩大到完整历史反问题父目录查找后，确认历史资料位于整理仓库的同级目录；此前“未找到历史权重”的判断只覆盖 `pinn-inverse-rotor-thickness` 子目录，范围不完整。

主要来源：`PINN_Inverse_fin_pls/models_final/` 中的 `upper_ring_bidirectional_final.pth`、`lower_ring_bidirectional_final.pth`、`rib_net_bidirectional_final.pth`；`thickness/` 中上下圆环的 `*_thickness_net_reg.pth` 与 `*_thickness_net.pth`。同时存在历史评估源 `pinn_mse_v3.py`、`Dataset/` 与 `Outer_Dataset/`。这些目录保持只读。

已添加 [tools/inspect_legacy.py](tools/inspect_legacy.py)，在本项目 uv 环境中严格加载权重，按历史评估源的网络变换执行分批预测及四指标计算。只提取已核对的网络/弧长定义，不执行旧脚本的顶层数据处理、训练、交互询问或写文件逻辑。无需重新训练；运行结果存入新的 `artifacts/<run_id>/`。

```powershell
$LegacySourceDir = Read-Host '历史资料目录，仅迁入外部资料时使用'
uv --cache-dir .cache/uv run --locked --extra cpu python tools/inspect_legacy.py --source-dir $LegacySourceDir --dataset-kind outer --run-id MY_LEGACY_OUTER
```

`--dataset-kind domain` 选择历史 `domain_data_*.csv`；默认 `outer` 选择 `T_d*_outer_long*.csv`。每种总体分别建立实验，不混合或调整划分来迎合旧报告。

| 记录 | 权重加载/推理 | 本次总体与 RMSE |
|---|---|---|
| `R004_LEGACY_REUSE` | 3 个温度、4 个厚度模型严格加载；预测有限 | 外表面温度表 603374 行；总体 6.2239 K |
| `R005_LEGACY_DOMAIN` | 同组权重，保持历史变换 | 域内温度表 397239 行；总体 1.8272 K |

两次结果都**未匹配**旧报告的 6.0609 K；旧报告总体为 1074640 点，样本量与本次资料不同。准确原因和旧评价数据版本仍需回查，不能据此宣称性能改善或精确复现。旧报告及当前数据/权重/评估源的哈希、网络张量形状和结果比较均保存在对应 `manifest.json`、`legacy_profile.json`、`metrics.json`；这不是新的独立测试集评价，也没有验证新全长接触模型的真实精度。

历史输入是圆环 `[r/0.225,theta,Fo]`、肋 `[s/0.225,Fo,rib_id]`，其中 `Fo=t*alpha_ref/0.225²`，输出门控为 `1-exp(-10*Fo)`，肋弧长从 0.120 m 起算。它们不同于新库的时间尺度、肋列顺序及 0.123 m 参考几何。历史厚度 `_reg` 使用原厚度列单位；非 `_reg` 训练输出经 `DELTA_REF` 归一化，该尺度没有保存在原始裸 state_dict 中，不能静默按 m 使用。历史 `.pth` 不能直接当作新 `models.pt/thickness.pt` 格式；统一 `rotor-pinn evaluate` 入口继续面向新库运行目录。这里的复用是按历史定义推理，尚未将旧参数转换成新模型或断点续训。

### 本地保留的三个版本

`R006_VERSION_SELECTION` 对 8 组定义已核对、具有上下圆环和肋完整权重的历史模型作同总体比较。固定使用 `PINN_Inverse_fin_pls/Dataset` 的外表面表，上环 188567 行、下环 265630 行、肋 142511 行；肋半径取所有候选共同范围 `[0.123,0.225]` m，总计 **596708** 行。没有重训；按相同样本的总体 RMSE 递增保留前三：

| 优先级 | 本地版本 | 原目录 | 本次比较 RMSE (K) |
|---|---|---|---|
| 1 | [fin_v3_1](docs/reference_models/fin_v3_1/profile.json) | `PINN_Inverse_fin_pls/models_v3.1` | 5.3087 |
| 2 | [fin_v3](docs/reference_models/fin_v3/profile.json) | `PINN_Inverse_fin_pls/models_v3` | 5.4839 |
| 3 | [fin_final](docs/reference_models/fin_final/profile.json) | `PINN_Inverse_fin_pls/models_final` | 6.2053 |

这里的“优先”限于已比较的 8 组完整转子模型及当前数据总体。旧数据可能参与过训练，不能解释为独立泛化排名；肋截取范围与 `R004` 不同，不能用数字差异宣称复现或改善。POD/MoE 等任务不同的模型、只有圆环的版本、重复权重和变换定义冲突的版本不加入这次排序；完整候选结果、排除原因、原权重形状/哈希与共同输入快照保存在 [R006](docs/verification/records/R006_VERSION_SELECTION/metrics.json)。原始 checkpoint 未记录训练源码提交，因此存档源码是定义匹配的参考快照，不声称恢复当年精确训练版本。

每组目录结构：

```text
fin_v3_1/
  profile.json              # SI 尺度、历史输入顺序、文件 SHA256、来源和选取依据
  factory.py                # 只含已核对的网络定义及显式导入
  weights/                  # upper/lower/rib.pth + 两个输出 m 的厚度权重
  examples/                 # 每域 195 条生成输入与参考预测，无需 COMSOL 数据
  source/                   # 原训练/预测/厚度源码快照
  verification.json         # 初次保存的加载、预测和重载检查
```

实际复用执行 `reuse_preserved.py` 或 `rotor_pinn.legacy.LegacyBundle`，保留的 `source/` 供阅读和溯源。旧训练脚本包含顶层 I/O 和历史路径，新的训练实验使用本项目 TOML 与 `rotor-pinn train`。历史裸权重不自动转成新全长接触网络，也没有用它们断点续训新模型。

进入本项目后，可直接验证推荐版本、运行 CSV 预测，或验证全部三个版本：

```powershell
uv --cache-dir .cache/uv run --locked --extra cpu python tools/reuse_preserved.py --bundle local_reference_versions/fin_v3_1 --verify
uv --cache-dir .cache/uv run --locked --extra cpu python tools/reuse_preserved.py --bundle local_reference_versions/fin_v3_1 --region upper --input local_reference_versions/fin_v3_1/examples/upper.csv --output artifacts/my_legacy_upper.csv --thickness
uv --cache-dir .cache/uv run --locked --extra cpu python tools/verify_preserved.py
```

预测 CSV 用 `r,theta,t`（m、rad、s），肋另需整数 `rib_id=0..12`；输出 `predicted_T` 为 K，`--thickness` 仅用于上/下环并输出 `predicted_thickness_m`。缺失文件、文件哈希不一致、非法输入或非正厚度会报错，已存在的输出文件拒绝覆盖。物理厚度沿用旧 `_reg` 权重的 m 输出；旧网络不保证全域严格正值，检查通过的范围见生成示例和历史检查记录。

Python 接口：

```python
import pandas as pd
from rotor_pinn.legacy import LegacyBundle

model = LegacyBundle("local_reference_versions/fin_v3_1")
frame = pd.read_csv("local_reference_versions/fin_v3_1/examples/rib.csv")
temperature_k = model.predict(frame, "rib")
```

`temperature_tensor(region, normalized_inputs)` 和 `thickness_tensor(region, normalized_radius_theta)` 保留坐标及参数自动微分，归一化定义见该版本 `profile.json`。调用者负责按物理尺度转换导数；推理加载默认 CPU/float32。加载只读取本地保留目录，`original_path` 等字段仅作溯源。

`V004_LEGACY_PORTABLE` 将三个目录复制到独立检查目录，并将历史来源路径置为不存在的路径：**15 个模型均严格加载**，每域生成示例预测有限，温度/厚度输入求导有限；重载预测最大差 **0 K**。完整 CPU pytest **37 通过、1 CUDA 跳过**，Ruff 通过；CUDA 硬件/自动微分验收沿用前述 `V002`，本轮历史移植检查使用 CPU。证据见 [V004](docs/verification/records/V004_LEGACY_PORTABLE/metrics.json) 和其 `full-pytest.xml`。

`local_reference_versions/` 已整体加入 `.gitignore`，权重、数据和运行产物仍保持本地。本次没有初始化、提交或推送 Git，也没有删除或改写父目录的历史版本。未来筛选使用 `tools/curate_legacy.py --source-root ... --keep 3 --output <新的目录>`；既有保留集和运行编号拒绝覆盖。

| 编号 | 日期 | 决策 | 依据/状态 |
|---|---|---|---|
| D001 | 2026-10-03 | 已知厚度先验下温度重建作为首版 | 用户批准计划；已采纳 |
| D002 | 2026-10-03 | 全长耦合、零界面阻力、半厚度等效阻力 | 用户选择及降维假设；已采纳 |
| D003 | 2026-10-03 | 解析肋几何及匹配求积 | 参考几何与守恒要求；已采纳 |
| D004 | 2026-10-03 | uv 独立环境、互斥 CPU/cu130、空间分组 | 用户批准计划；已采纳 |
| D005 | 2026-10-03 | 以代码与模型可用为验收，保留 3 组可比较历史模型，只存本地 | 用户放宽精确指标复现要求；已采纳 |

## 一轮工作与隐私

读 README、AGENTS 和有关假设/实验行，先登记目的、数据版本、预算和评价，执行独立 run_id，保留失败、异常和负结果。计算正确性与研究解释分别判断；关键路线改变由负责人确认；作者视角由作者维护。

受限原数据、输入快照、权重、缓存和运行产物默认忽略提交；合成函数和方法代码可复用。发布前人工核查权限、署名和完整拟提交内容。本次只创建本地项目，没有发布数据、安装 Hook 或修改原仓库。


## 研究材料整理

[模型、现有推导与缺口](docs/theory/README.md)汇合代码目录、毕业设计、历史备份和文献中的相关材料，保留仓库内整理稿、来源编号与哈希。本轮只整理既有内容并列出缺口，未补写证明、重新训练或调整假设支持状态。详见本项目资料入口。

## 开源许可与集合布局

[English](README.md) · [MIT](LICENSE) · [贡献说明](CONTRIBUTING.zh-CN.md)

公开范围内的项目自有代码与原创整理稿采用 MIT；依赖保留原许可，历史数据、模型与第三方原件不纳入本次发布授权。迁移后的 uv 环境按原锁文件重建，科学假设状态不变。
移动后实际检查：[MOVE20261004_01](docs/publication/relocation.json)。
