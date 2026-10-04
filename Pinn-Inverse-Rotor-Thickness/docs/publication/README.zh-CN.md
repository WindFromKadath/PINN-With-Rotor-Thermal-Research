# 发布范围

[English](README.md)

本项目已归入 Rotor-Thermal-Research 集合。以后以集合目录作为 Git 根目录，外层本地项目集合不纳入 Git。数据与配置仍相对子项目根目录解析，uv 环境独立维护；也可单独复制使用。

代码、配置、锁文件、实现测试、必要原创整理稿和脱敏来源元数据可按本地 MIT LICENSE 准备提交。原始数据、网格、权重、历史代码包、第三方原件、产物、环境与缓存保持本地并忽略。索引只记录来源，不授予再发布权限。

在子项目根目录执行：

```powershell
uv run --locked --extra cpu python tools/check_git_ready.py
```

[候选检查](git_readiness.json) · [清单](manifest.json) · [既有干净副本检查](clone_checks.json) · [无私有资产的运行边界](../running_from_clone.md)

MOVE20261004_01 只变更位置与发布布局，旧提交范围决定保留为历史记录。物理公式与假设状态未改变；未初始化、提交或推送 Git。候选扫描之后，实际发布前仍应检查暂存内容、署名和历史。
