# 贡献说明

[English](CONTRIBUTING.md)

说明影响的配置、预期行为，使用合成或允许公开的数据给出最小示例。反馈中去掉私人路径、凭据和保密原始日志。以后配置 GitHub 远端时使用对应 Issue/PR；本地开发无需远端。

使用相关子项目的锁定 uv 环境，在子项目中执行相关测试。历史模型测试需本地资产，缺失时不得写成通过。在本页所在目录可执行以下检查：

```powershell
uv run --directory Pinn-3D-Thermo-Mechanical-Rotor --locked --extra cpu python ../tools/check_publication.py
uv run --directory Pinn-3D-Thermo-Mechanical-Rotor --locked --extra cpu python ../tools/check_publication.py
```

物理改动与目录/发布改动分开；新实验使用独立编号并记录条件与失败。公开中英文命令、接口、版本和限制同步，原始研究笔记可以保持中文。

新增项目代码与原创文本采用 [MIT](LICENSE)。仅提交有权按 MIT 许可的材料，兼容的复用内容保留归属；安装的依赖继续遵循各自条款。不提交原始数据、第三方论文、历史来源包、权重、环境或审计日志。提交前检查实际暂存内容与署名，发布前另检查历史与附件；上述命令不批准推送。

仓库通过 `.gitattributes` 保留原文件字节和换行形式，使已记录的 SHA-256 在不同平台仍可核对。改变换行时应同步相关来源清单。
