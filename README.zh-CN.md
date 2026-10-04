# 四方向转子热学研究参考库

[English](README.md)

四个可复用 Python 参考库，共同维护代码、必要理论整理、来源索引与有限验证结果。各方向保留自己的物理假设、尺度和模型变换。当前验收是代码与模型可用；迁移检查不证明历史精度、收敛性或替代 COMSOL 的能力。

| 目录 | 方向 | Python | PyTorch |
|---|---|---|---|
| [Neural-Operator-Reference](Neural-Operator-Reference/README.zh-CN.md) | 算子学习、圆环 PINO 与 PI-DeepONet | 3.13.14 | 2.14.1 |
| [Pinn-Inverse-Rotor-Thickness](Pinn-Inverse-Rotor-Thickness/README.zh-CN.md) | 已知厚度温度重建；未知厚度反演仍有研究缺口 | 3.12.14 | 2.12.1 |
| [Pinn-Ring-Rib-2.5D](Pinn-Ring-Rib-2.5D/README.zh-CN.md) | 加热/冷却环肋温度 PINN 与耦合流程 | 3.12.14 | 2.12.1 |
| [Pinn-3D-Thermo-Mechanical-Rotor](Pinn-3D-Thermo-Mechanical-Rotor/README.zh-CN.md) | 3D 瞬态热学 → 轴对称准静态热弹性单向耦合 | 3.13.14 | 2.14.1 |

## 安装与检查

在集合根目录执行以下命令，需先具备 uv。各子项目独立维护锁文件、Python 版本、环境和本地缓存，不合并为 uv workspace。cpu/cuda extras 互斥；CUDA 安装按各子项目说明进行。

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

导入模块不会启动训练。配置、合成示例与数据相关流程见各子项目说明；必要理论整理已经接入仓库，历史笔记以中文为主。

## 提交内容与本地资产

Git 候选范围含代码、配置、锁文件、测试、必要整理稿和脱敏元数据。原始 COMSOL 表、网格、权重、精选历史代码包、完整来源文档、运行产物、环境与缓存保持本地并忽略。新克隆能执行已说明的基础检查与合成流程；历史重放需另恢复本地资产。忽略目录中的文件不在本次 MIT 发布授权范围内。

[研究阅读入口](docs/research.md) · [发布范围与来源](docs/publication/README.zh-CN.md) · [实际检查记录](docs/publication/readiness.json) · [贡献说明](CONTRIBUTING.zh-CN.md)

## 许可

公开范围内的项目自有代码与原创整理文本仅采用 [MIT](LICENSE)，四个子项目持有同一许可副本。安装的依赖保留自己的许可证。第三方原文、数据和权重排除；索引、引用不授予再发布权限。含非商业限制的上游 DeepONet 文件未迁入实现库；来源限制继续见各子项目说明。
