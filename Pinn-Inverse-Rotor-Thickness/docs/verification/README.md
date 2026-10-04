# 实现验收的仓库内摘要

CPU 37 项通过、1 项 CUDA 跳过；V004三个包15模型严格加载，温度/厚度导数有限，重载温度差为零。GPU的已有加载/导数检查与CPU测试分开记录。

[主要记录](records/V004_LEGACY_PORTABLE/metrics.json) · [研究证据与缺口](../theory/evidence_and_gaps.md)

这是2026-10-03/04既有迁移检查的整理，原始记录哈希见[接入追踪](integration_provenance.json)。全量数据、旧权重、日志与训练产物未加入仓库；当前文本接入没有重跑历史精度或长预算训练。数值可用与实现一致性不能证明物理近似、泛化、收敛或未知参数辨识。

新克隆的安装与检查边界见[运行说明](../running_from_clone.md)。
