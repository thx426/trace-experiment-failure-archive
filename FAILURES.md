# 失败、负面发现与修复记录

下表及案例是对已保存记录的索引。区分运行故障、验证错误、方法风险和效果限制；没有将所有问题都解释为最终分数的原因。

| 编号 | 阶段 | 问题 | 记录中的结果 |
|---|---|---|---|
| F01 | 8 月历史实验审计 | 配置、源码与实际运行证据不齐全 | 保留可验证的局部结论，不追认缺失运行 |
| F02 | 10 月信息依赖审计 | 指导混有任务实例知识与评分知识；字面 ID 检查不充分 | 增加保守 gate；仅 CPU 验证，未证明性能改善 |
| F03 | 10-24 compact 首次启动 | CUDA runtime 选择错误 | 初始化失败，无 rollout、无优化器更新；修复环境后另开目录 |
| F04 | 10-24 WebShop R2 | 过载、非幂等重试及终局响应恢复失败 | 保存旧现场，修复协议与服务分片，R3 重新启动 |
| F05 | 10-24 → 10-25 Skill 产出 | 320 组中 170 组没有 Skill | 修复收尾预算、上下文提醒和失败处理，进入新方法版本 |
| F06 | 10-25 Clean v1 | 属性解析遗漏；验证复用同一解析器导致误判 | 停止 v1，增加独立 oracle，v2 从原始起点启动 |
| F07 | 10-25 Clean v2 最终评测 | 整体改善仍伴随逐题退步与固定分布局限 | 保留 10 道完全丢失题及行为分析，不仅报告均分 |
| F08 | 10-23 循环推理机制实验 | 可读出的隐状态不能直接作为下一轮可用输入 | 在小模型受控任务上观察失败；训练适配可修复，不能外推为普遍结论 |

## F01：历史实验不能只靠方法名和配置认定

早期实现包含 OPD、OPD+GRPO、OPD+SFT；后期 classic PGS 为采样引导与不对称分臂优势更新，不能统称 OPD，也不能把后期行为倒填到早期运行。22 份可解析 launch manifest 与实际命令一致，另有零字节 manifest。71 份本地 `main_ppo.log` 为空，不构成运行证据。

SpreadsheetBench 的一段现存日志能验证 step 101–167 共 67 个更新步的 actor loss、pg loss、梯度和引导权重，不能替代全部 run 的完整证据。10 月 pilot 的 14 步、7168 条轨迹也不能充当 8 月的历史实验数据。

证据：[训练目标审计](docs/training-objective-audit-20261023.md)、[历史机制审计](docs/grpo-mechanism-audit-20261023.md)、[时间范围更正](reports/grpo_mechanism_20261023/README.md)。

## F02：没有提前复制 ID，不代表不存在信息依赖

10 月 pilot 的 319 份非空文档中，294 份含商品 ID。按时序检查未发现直接提前使用文档 ID，但部分指导包含“无需检查某属性”“评分接受某解释”等超出当前学生观察的知识。成功或正优势不能自动证明指导可迁移。

初版只匹配大写 ID，漏掉小写，统计从 194 修订为 294。后续还更正了初始可见信息的范围。增加的字面定位符 gate 默认关闭，仅完成 CPU 测试与 dry-run；不是语义泄漏检测器，也没有新训练改善证据。该 pilot 不覆盖原先后期下降的完整运行，不能据此宣布已找到下降的唯一原因。

证据：[信息依赖审计](docs/skill-information-dependency-audit-20261023.md)、[摘要](reports/skill_information_20261022/summary.json)、[案例摘录](reports/skill_information_20261022/case_excerpts.json)。

## F03：训练开始前的 CUDA 初始化失败

compact3072 首次启动在 SGLang preload 阶段选中系统 CUDA 运行库，缺少 `cudaGetDriverEntryPointByVersion`。记录确认没有 rollout 或优化器更新。修复学生子进程的 CUDA runtime 优先级后在 R2 新目录启动，不能把失败目录与新运行拼成一条轨迹。

证据：[compact 实验执行记录](reports/compact_pgs_20261024/README.md)。

## F04：WebShop 传输失败与服务负载

R2 只留下 202 条未完成的初始评测记录，优化器更新为零。已确认的实现问题包括全局锁、默认 backlog=5、聚合高并发、对非幂等修改操作的重试，以及终局购买响应丢失后无法恢复。成功响应的 BrokenPipe 还可能触发二次写入。

修复包括稳定 reset UUID、单调 action sequence 与结果缓存、release tombstone、限定重试类型、故障显式中止、分片服务与队列计时。记录中的 409/500/BrokenPipe 数量覆盖旧共享服务生命周期，不能全部归到 R2；旧服务没有逐请求时间证据，无法重建第一条慢请求。吞吐改善也不能单独归因为研究员并发配置或算法改动。

证据：[修复报告](reports/webshop_repair_20261024/repair_report.json)、[CPU 验证](reports/webshop_repair_20261024/cpu_validation.json)。

## F05：没有研究异常，也可能没有 Skill

R3 前 20 步共 320 组研究会话，170 组没有 Skill，但记录的 `research_error=0`：146 组未调用 finalize，15 组超长改稿被拒绝，9 组返工后未在预算内完成。compact 重建输入漏带结束提醒、末轮仍能反复读取原始轨迹、拒绝超长稿后缺少明确产出处理共同构成实现缺陷。

新版本保留收尾轮次、限制重复读取、编译短检查表、区分 researcher_finalized 与 fallback，并对基础设施失败显式中止。离线回放覆盖旧缺失组只证明实现契约，不能直接证明学生能力提高。这是方法版本变化，不能仅称为提速。

证据：[Clean-Skill 执行记录](reports/skill_clean_20261025/execution_report.txt)。

## F06：自检复用生产解析器，无法发现自己的错误

Clean v1 只识别 color、size、fit type，将 `item shape: round` 合并到颜色字段。最初的零差异核验复用了生产解析器，结论不可靠，修复记录明确否定了该项初检结论。对训练 / dev 的公开指令检查分别发现 15 / 14 题存在遗漏或合并。

停止 v1 时没有完成的优化器更新，但已经花费 16 个研究会话、191497 个研究员 token，并完成 1280 条初始评测和 128 条首批 rollout。增加独立规格 oracle、按属性边界解析全部标签并对无法解析的情况报错后，CPU 测试与公开规格检查通过；v2 从原始 checkpoint 启动。v1 的调用与成本不能计入 v2 成功运行。

证据：[v1 复盘与 v2 修复](reports/skill_clean_20261025/repair_v2/repair_report.txt)。

## F07：最终均分改善没有消除逐题退步

第 100 步 Clean-Skill v2 对比 GRPO：58 题成功次数提高，18 题下降，180 题相同；19 题从 0/5 变为有成功，同时 10 题从有成功变为 0/5。可见的失败包括漏选 `item shape`、简化查询时删掉区分特征、目标选项不存在仍购买。

两组均未打开 Description/Features，且高度依赖首个搜索候选。212/256 道 dev 题的规格前文字前缀出现在训练中。研究员另有 1600 个会话、3502 次探索实验、7511 条额外 episode；这不是总计算预算匹配的比较。只有一个训练 seed，dev 被用于人工分析和停止规则，没有独立保留 test。逐题配对不确定性不等于多训练 seed 的不确定性。

证据：[最终逐题复盘](reports/skill_clean_20261025/step100_comparison/final_review_zh.txt)、[完整比较摘要](reports/skill_clean_20261025/step100_comparison/summary_complete.json)。

## F08：隐状态能读出答案，不保证可以继续计算

约两万参数的小模型在单步任务上学会读出正确符号后，直接反馈原隐状态的多跳正确率下降。输入输出权重绑定时，第二跳为 81.85% ± 17.66%，第 16 跳为 20.10% ± 6.28%；显式符号反馈和软重嵌入在该设置上均为 100%。循环训练后可修复此任务的接口问题。

这些是三个训练 seed 的受控机制结果，后续适配增加了训练计算；不是大模型榜单，也不是“隐式推理天然更差”的结论。完整参数、预测核验、图表及未绑定输出头的敏感性结果见原记录。权重文件不在此归档。

证据：[实验设计与结果](reports/loop_transformer_20261023/README.md)、[实测表](reports/loop_transformer_20261023/measured_table.md)、[核验记录](reports/loop_transformer_20261023/verification.json)。
