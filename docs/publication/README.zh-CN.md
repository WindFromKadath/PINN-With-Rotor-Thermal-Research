# 发布范围与来源

[English](README.md)

四个子项目移入本集合，未导入 Git 历史。子项目自有代码、锁文件、测试和原创整理稿是预定公开内容，仅采用 MIT；集合与子项目根目录保留相同 LICENSE。外层父目录及第五个无关项目不纳入范围。

各项目来源清单用相对路径和哈希绑定历史材料；清单记录来源，不转移权利。算子实现明确排除了含非商业限制的上游 DeepONet utils.py、ADR_solver.py。依赖经 uv 安装，继续采用自己的许可；其实现文件未作为依赖源码内嵌。历史代码包、论文、COMSOL 表、网格和权重继续排除。不能将全部依赖宣称为 MIT，也不能把本许可用于第三方原件。

[算子来源](../../Neural-Operator-Reference/README.zh-CN.md) · [反问题来源](../../Pinn-Inverse-Rotor-Thickness/README.zh-CN.md) · [环肋来源](../../Pinn-Ring-Rib-2.5D/README.zh-CN.md) · [热力来源](../../Pinn-3D-Thermo-Mechanical-Rotor/README.zh-CN.md)

## 检查与限制

[当前候选检查](readiness.json)和[候选哈希](manifest.json)覆盖公开文本/代码；移动证据保存在各子项目忽略的 artifacts 中。原配置、科研源码、测试和 uv 锁文件不变；CPU 环境在新路径重建，旧环境目录保留并忽略。本次 CPU/模型结果写入检查报告，与旧科研实验分开。

本次不建立 Git 仓库或远端，不提交、推送、安装 Hook 或改变署名。实际发布前应按独立协作模板库中的通用 MIT/发布流程，检查暂存内容、署名、拟发布分支历史和附件。当前仓库的运行与必要理论不依赖访问模板库。候选检查不替代专门密钥扫描、图像检查或对以后新增材料的权利核验。
