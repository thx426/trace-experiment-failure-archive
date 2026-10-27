# 实验索引与保存的结果

记录时间范围以文件内容为准。配置、部署、CPU 测试、GPU rollout 与训练完成属于不同证据层级。

| 阶段 | 内容 | 证据入口 |
|---|---|---|
| 2026-08 classic | ALFWorld / WebShop / SearchQA / Math / SpreadsheetBench；部分运行证据缺失 | [历史审计](docs/grpo-mechanism-audit-20261023.md)、[run 矩阵](reports/training_objective_20261023/run-matrix.csv) |
| 2026-10-18 至 23 pilot 与机制审计 | 训练目标、信息依赖、实际优势与入训链路；不可充当 8 月数据 | [目标审计](docs/training-objective-audit-20261023.md)、[信息审计](docs/skill-information-dependency-audit-20261023.md) |
| 2026-10-23 pass5 | GRPO / All-Skill 等对照及逐题行为分析 | [分层结果](reports/grpo_vs_allskill_stratified_20261024/analysis.json)、`reports/pass5_20261023/` |
| 2026-10-24 compact 与服务修复 | CUDA 初始化、R2 失败、R3 服务修复及缺 Skill 诊断 | [compact 记录](reports/compact_pgs_20261024/README.md)、[传输修复](reports/webshop_repair_20261024/repair_report.json) |
| 2026-10-25 Clean v1 → v2 | v1 规格缺陷中止；v2 独立解析审计及 100 步 bare 评测 | [修复记录](reports/skill_clean_20261025/repair_v2/repair_report.txt)、[最终审计](reports/skill_clean_20261025/step100_comparison/final_review_zh.txt) |
| 2026-10-23 Loop Transformer | 小模型受控多跳机制实验，三个训练 seed | [设计、结果与限制](reports/loop_transformer_20261023/README.md) |

## 方法身份

早期 OPD 的教师分布匹配损失、后期 classic PGS 的采样引导与不对称优势，以及 Clean-Skill 的普通 GRPO 更新必须分开描述。Clean-Skill v2 不是历史 asymmetric PGS，也没有 OPD/SFT 损失。见 [训练目标审计](docs/training-objective-audit-20261023.md) 和 [最终方法说明](reports/skill_clean_20261025/step100_comparison/final_review_zh.txt)。

## Clean-Skill v2 对比 GRPO 的 bare dev 曲线

256 题 × 5 次；pass@1 估计为成功 rollout 数 / 1280，pass@5 为至少一次成功题数 / 256。下表以百分比表示。

| Step | GRPO pass@1 | Clean pass@1 | GRPO pass@5 | Clean pass@5 |
|---:|---:|---:|---:|---:|
| 20 | 20.703125% | 23.125000% | 50.781250% | 57.421875% |
| 40 | 55.468750% | 69.140625% | 76.171875% | 80.078125% |
| 60 | 66.562500% | 74.765625% | 78.906250% | 82.812500% |
| 80 | 72.812500% | 77.812500% | 80.468750% | 82.421875% |
| 100 | 73.203125% | 79.609375% | 80.078125% | 83.593750% |

结构化来源：[summary_complete.json](reports/skill_clean_20261025/step100_comparison/summary_complete.json)。完整数值以分子、分母为准。

## 不把改善解释过头

第 100 步增加 82 次成功和 9 道至少一次成功题，但仍有 18 题成功次数下降、10 题完全丢失。第 20 步相对于 GRPO 丢失的 28 题到最终均恢复，这不代表所有负迁移消失。

训练 seed 只有 42；dev 被用于中途分析与停止规则；研究员有额外成本；未做完整消融，未匹配总计算预算。bootstrap 只反映题目抽样不确定性。查询简化既有受益题也有受损题，不能直接推出“越短越好”。

完整限制和逐题动作见 [最终复盘](reports/skill_clean_20261025/step100_comparison/final_review_zh.txt)。
