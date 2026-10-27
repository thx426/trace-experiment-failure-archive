# TRACE 实验与失败记录归档

Research logs, negative results, repairs, and audited evidence for TRACE / OPD / PGS / Clean-Skill.

本仓库公开一个研究项目在 **2026 年 8 月至 10 月**留下的本地实验材料：实验配置、启动记录、诊断、失败现场、修复说明、逐题比较及审计结果。整理日期为 **2026-10-28（Asia/Shanghai）**。它是一份经过脱敏的历史快照；各文件中的“当前”“已启动”等措辞均指记录当时，不表示仓库发布时仍在运行。

失败并不等于整个项目没有结果。这里同时保留中断、无效验证、负面发现与后续改善，避免只展示最终分数。归档没有重跑训练，也没有补造缺失证据。

## 从哪里开始

- [失败与修复复盘](FAILURES.md)：症状、已确认问题、处理、证据与未解决边界。
- [实验索引与结果](EXPERIMENTS.md)：方法身份、阶段和 Clean-Skill / GRPO 对照曲线。
- [证据范围、脱敏与复核](REPRODUCIBILITY.md)：哪些文件公开、哪些排除、如何核验文件与指标。
- [文件清单](MANIFEST.json)：每份记录的源文件与公开文件 SHA-256、字节数、脱敏类型与次数。
- [排除清单](EXCLUSIONS.json)：被考虑但未发布的文件及原因；隐藏环境目录未遍历。

## 一个有结果、也有明确限制的对照

保存的第 100 步审计记录：同一 Qwen2.5-3B 学生起点、seed=42、同一 256 道 dev 题、每题 5 次、temperature=0.6，评测时均无 Skill。

| 方法 | 成功次数 / 1280（pass@1 估计） | 至少一次成功题数 / 256（pass@5） |
|---|---:|---:|
| 普通 GRPO | 937 / 1280 = 73.203125% | 205 / 256 = 80.078125% |
| Clean-Skill v2 | 1019 / 1280 = 79.609375% | 214 / 256 = 83.59375% |
| 差值 | +6.40625 个百分点 | +3.515625 个百分点 |

证据：[完整 checkpoint 比较](reports/skill_clean_20261025/step100_comparison/summary_complete.json)、[最终逐题审计](reports/skill_clean_20261025/step100_comparison/final_review_zh.txt)。这是单训练 seed、被用于分析及停止规则的 dev 结果；研究员还有额外成本，未匹配总计算预算。pass@5 增益的题目配对 bootstrap 区间跨零；不能将其写成独立 test 上已证实的普遍优势。未做 A/B/C/D 消融，不能将提升唯一归因于某条清洗规则。

## 目录

| 目录 | 内容 |
|---|---|
| `docs/` | 方法身份、训练目标、机制与信息依赖审计、运行手册 |
| `reports/` | 分阶段实验记录、修复复盘、统计表、逐题比较与离线分析脚本 |
| `runs/` | 8 月 classic 实验的现存配置、launch manifest、日志 |
| `configs/` | 本地留存的实验配置；配置存在不等于运行完成 |
| `probe/` 与根目录分析脚本 | 历史诊断脚本，仅供阅读；部署字段已脱敏 |
| `tools/` | 归档构建脚本和只读完整性验证脚本 |

原 `outputs/` 中没有符合范围的非空文件。模型权重、第三方数据集、虚拟环境、二进制部署包、运行截图、私人协作文档、聊天历史、论文原稿及第三方源码快照不包含在本仓库。被省略的文件路径及原始记录中引用的远端路径可能仍用于说明证据出处，不能据此认为相应文件已公开或可下载。

## 使用

```bash
git clone https://github.com/thx426/trace-experiment-failure-archive.git
cd trace-experiment-failure-archive
python3 tools/verify_archive.py
```

只读验证使用 Python 标准库，不调用模型、云端或训练服务。历史脚本并不是完整可执行的训练发行版；请先读 [复核说明](REPRODUCIBILITY.md)。

## 许可与引用

原创代码和作者撰写的说明使用 [MIT License](LICENSE)。第三方内容及模型输出的范围见 [NOTICE](NOTICE.md)，不由此获得额外授权。

引用时请给出仓库 URL、所用 commit，以及具体证据文件路径；数字与结论应绑定记录的日期、方法版本、checkpoint 和评测口径。此归档没有指定论文题目、作者名单或 DOI。
