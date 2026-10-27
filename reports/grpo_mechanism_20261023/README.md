# 证据范围更正

本任务的“历史实验”指项目 `runs/classic_*` 中 **2026 年 8 月的实验**。主结论见 `august/audit.json`、`august/matched-checkpoints.csv` 及 `docs/grpo-mechanism-audit-20261023.md`。

`skill-info-compact-20261023.json.gz`、`replay/`、`replay.log` 来自 **10 月 API 试跑**；它们不是 8 月实验数据，不用于本任务的历史效果判定。保留这些文件用于追溯此前分析，不能把其中 14 步、7168 条轨迹、缺文档比例等数值写成 8 月结果。

`two_action_counterexample.json` 和 `august/audit.json` 中的 synthetic_controls 是当前生产函数的 CPU 合成对照，不是模型训练实验。`tests.txt` 是本地回归测试结果。

已按用户要求停止云端操作。GPU 由另一线程部署和使用；本线程没有启动研究员或学生训练，也不调度 GPU。`august/experiment-plan.json` 仅为待执行方案，不会自动运行。

本线程新增的 `steering.fallback_advantage` 默认值为 `assigned`，保留旧行为；`effective_bare` 仅为显式启用的候选消融。它尚无新训练效果证据。共享工作区中另外出现的 task gate、模型下载和部署文件不归属于本轮实现成果。
