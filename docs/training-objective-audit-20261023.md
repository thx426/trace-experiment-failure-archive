# GRPO 与 OPD 训练目标核对

核对日期：2026-10-23。范围：本地 `TRACE-OPD-TERM-minimal` 的训练实现、11 个 run 的启动材料和现存日志，以及 PGGRPO 历史版本。本文区分源码、启动配置和实际更新日志三个证据层级；未重跑训练。本文源码行号链接到审计时冻结的副本；本轮结束前检测到工作区另外加入 task gate 的改动，已单独记录在 [source-drift.patch](https://github.com/thx426/trace-experiment-failure-archive/blob/main/reports/training_objective_20261023/source-drift.patch)，不将其倒填到历史配置。

**结论：两代方法必须分开写。后期 classic 基线为关闭组内标准差归一化的 GRPO；PGS 为采样引导加不对称分臂优势更新，没有 OPD 损失。早期仓库确实实现并配置了纯 OPD、OPD+GRPO 和 OPD+SFT，不能因为现在删掉了蒸馏代码，就说项目从未使用 OPD。历史各 run 是否按对应配置完成训练，仍需其原始日志闭环。**

## 1 当前实验的身份

| 实验组 | 启动参数与代码 | 准确名称 | OPD |
|---|---|---|---|
| `classic_alf3b_grpo`、`classic_alf7b_grpo`、`classic_ws3b_grpo`、`classic_sqa7b_grpo` | `algorithm.adv_estimator=grpo`；`norm_adv_by_std_in_grpo=False` | 不做优势标准差归一化的 GRPO 基线 | 没有启用蒸馏项 |
| 对应 `*_pgs`，另含 SQA legacy、Math、SpreadsheetBench，共 7 个 run | `pgs_asym_grpo`；`steering.enabled=true`；`steered_fraction=0.5`；`w_beh_mode=off` | 采样引导与不对称分臂优势更新的 GRPO 系列变体 | 没有启用蒸馏项 |

22 份可解析 launch manifest 的配置与实际 `cmd` 中的优势估计器一致，均未出现 `distillation.*` 启动项；另有 1 份零字节 manifest，不能作为有效启动证据。两组均关闭 `use_kl_in_reward` 和 actor 的 reference KL loss，采用 `seq-mean-token-mean` 聚合。后者先对各轨迹的有效 token 求均值，再按轨迹聚合；仅关闭优势标准差归一化不足以宣称完整复现 Dr. GRPO。

具体清单见 [run-matrix.csv](https://github.com/thx426/trace-experiment-failure-archive/blob/main/reports/training_objective_20261023/run-matrix.csv)，原始 WebShop 启动证据见 [PGS manifest](https://github.com/thx426/trace-experiment-failure-archive/blob/main/runs/classic_ws3b_pgs/launch_manifest_20260817_002909.json) 与 [GRPO manifest](https://github.com/thx426/trace-experiment-failure-archive/blob/main/runs/classic_ws3b_grpo/launch_manifest_20260817_002916.json)。manifest 未绑定源代码 revision，不能由配置单独证明整个历史运行的每一步行为。

## 2 当前实际训练信号

训练链路为：环境判分 → `traj.won` / `traj.score` → `AgentLoopOutput.reward_score` → `rm_scores` → `token_level_rewards` → 组内优势 → clipped policy loss。

- agent_loop.py:1026（原始源路径 `reports/training_objective_20261023/current_source_snapshot/src/trace_opd/train/agent_loop.py#L1026`；该副本未分发，见 EXCLUSIONS.json）：`reward_from_score=false` 时取成功指示量；为 true 时取环境连续分数。当前默认值是 false；历史 22 份 manifest 均未记录此字段，不能将今日默认值单独当作历史 runtime 的最终证明。
- AgentLoopOutput.as_dict:131（原始源路径 `reports/training_objective_20261023/current_source_snapshot/verl/verl/experimental/agent_loop/agent_loop.py#L131`；该副本未分发，见 EXCLUSIONS.json）：终局标量只写入 `rm_scores` 最后一个位置。环境反馈 token 是否参加损失由 response mask 决定。
- trainer_base.py:1517（原始源路径 `reports/training_objective_20261023/current_source_snapshot/verl/verl/trainer/ppo/v1/trainer_base.py#L1517`；该副本未分发，见 EXCLUSIONS.json）：关闭 reward KL 后，`token_level_rewards=token_level_scores`；PGS 走显式分支，不能只查看旧 `ray_trainer.py` 推断执行路径。
- core_algos.py:270（原始源路径 `reports/training_objective_20261023/current_source_snapshot/verl/verl/trainer/ppo/core_algos.py#L270`；该副本未分发，见 EXCLUSIONS.json）：GRPO 对整条响应的奖励求和、按任务分组求优势，再广播至有效响应 token。
- losses.py:101（原始源路径 `reports/training_objective_20261023/current_source_snapshot/verl/verl/workers/utils/losses.py#L101`；该副本未分发，见 EXCLUSIONS.json）：使用 native `vanilla` clipped policy loss；没有 teacher logits、teacher targets 或 distillation loss 输入。日志中的 `ppo_kl` 是策略更新诊断，不等于教师蒸馏 KL。

令同任务裸轨迹集合为 B，引导轨迹集合为 S，终局奖励为 R。对于有至少两条裸轨迹的正常组，当前 PGS 使用：

\[
A_i^{\mathrm{bare}}=R_i-\frac{1}{|B|-1}\sum_{j\in B,j\ne i}R_j,
\qquad
A_i^{\mathrm{steered}}=\alpha_t\max(R_i-\bar R_B,0).
\]

两臂不做标准差归一化。当前 classic 配置的 steered cap 为 0，未启用正优势总量上限。源码还处理孤立引导组、单条裸样本和 padding，详见 compute_pgs_asym_outcome_advantage:721（原始源路径 `reports/training_objective_20261023/current_source_snapshot/verl/verl/trainer/ppo/core_algos.py#L721`；该副本未分发，见 EXCLUSIONS.json）。

基线 GRPO 无标准差归一化时使用 `R_i−mean(group)`；PGS 裸臂使用 bare-only leave-one-out。两者的基线、尺度和样本分布均不同，不能把 PGS 与基线的全部差异归因于采样核。

\(\alpha_t=\alpha_0[1+\cos(\pi t/T)]/2\)，到 T 后为零，作用于**引导臂优势**；arms.py:29（原始源路径 `reports/training_objective_20261023/current_source_snapshot/src/trace_opd/steering/arms.py#L29`；该副本未分发，见 EXCLUSIONS.json） 同时控制引导臂退场。它不是 token 采样强度的余弦退火。`w_beh_mode=off` 时 agent_loop_tq.py:271（原始源路径 `reports/training_objective_20261023/current_source_snapshot/verl/verl/trainer/ppo/v1/agent_loop_tq.py#L271`；该副本未分发，见 EXCLUSIONS.json） 不装配该行为校正权重；因此也不能称为对引导分布作了完整校正的无偏 on-policy GRPO。

“教师没有直接损失项”不等于“教师不影响学习”：hook.py:491（原始源路径 `reports/training_objective_20261023/current_source_snapshot/src/trace_opd/steering/hook.py#L491`；该副本未分发，见 EXCLUSIONS.json） 用文档前缀评分，采样核和 SPS 改变最终轨迹；引导轨迹再通过奖励和优势进入更新。这属于采样侧指导，不是直接匹配教师分布的 OPD。

## 3 早期 OPD 在哪里实现

从固定历史提交 [4bc0da93（2026-07-26）](https://github.com/wus2000/PGGRPO/commit/4bc0da93f4a95cc6ba508cd119af0da729a0277e) 恢复了训练入口、数据流、蒸馏损失和实验配置；另从 [5beecec5（2026-08-02）](https://github.com/wus2000/PGGRPO/commit/5beecec57ba95503e4e7138317ed5ae2271bb9d4) 恢复后续 reverse-KL 实现。18 个文件均通过 Git blob SHA-1 校验。源码副本分别保存在 `reports/training_objective_20261023/legacy_4bc0da93/` 与 `legacy_5beecec5/`，不属于当前训练入口。

| 环节 | 历史实现位置（4bc0da93） | 已核实行为 |
|---|---|---|
| 启用蒸馏 | scripts/launch_train.py:237（原始源路径 `reports/training_objective_20261023/legacy_4bc0da93/scripts/launch_train.py#L237`；该副本未分发，见 EXCLUSIONS.json） | `distillation.enabled=True`、`self_teacher=True`、`forward_kl_topk`、K=64；`use_policy_gradient=False`、`use_task_rewards=False` |
| 学生采样与教师上下文 | agent_loop.py:351（原始源路径 `reports/training_objective_20261023/legacy_4bc0da93/src/trace_opd/train/agent_loop.py#L351`；该副本未分发，见 EXCLUSIONS.json）、同文件:554（原始源路径 `reports/training_objective_20261023/legacy_4bc0da93/src/trace_opd/train/agent_loop.py#L554`；该副本未分发，见 EXCLUSIONS.json） | 学生裸采样；同一响应 token 序列配上带研究文档的 teacher prefix；不是用 researcher 完整轨迹替代学生样本 |
| 教师打分 | teacher_topk_fsdp.py:106（原始源路径 `reports/training_objective_20261023/legacy_4bc0da93/verl/verl/trainer/distillation/teacher_topk_fsdp.py#L106`；该副本未分发，见 EXCLUSIONS.json）、engine_workers.py:714（原始源路径 `reports/training_objective_20261023/legacy_4bc0da93/verl/verl/workers/engine_workers.py#L714`；该副本未分发，见 EXCLUSIONS.json） | 同一 actor 的 FSDP 权重，在带文档上下文 `no_grad` 前向；输出 `teacher_ids` / `teacher_logprobs`。外部 researcher 负责写文档，不是提供该项 logits 的大模型教师 |
| token 级蒸馏信号 | distillation/fsdp/losses.py:75（原始源路径 `reports/training_objective_20261023/legacy_4bc0da93/verl/verl/trainer/distillation/fsdp/losses.py#L75`；该副本未分发，见 EXCLUSIONS.json） | 在教师 top-K 支持上计算教师加权的 log-prob 差；学生裸上下文前向接收梯度，教师目标不反传 |
| 聚合及反传 | distillation/losses.py:398（原始源路径 `reports/training_objective_20261023/legacy_4bc0da93/verl/verl/trainer/distillation/losses.py#L398`；该副本未分发，见 EXCLUSIONS.json） | 按 response mask 聚合并直接反传；该配置不是把负 KL 作为奖励再跑 GRPO |
| 组合 GRPO | distillation_ppo_loss:177（原始源路径 `reports/training_objective_20261023/legacy_4bc0da93/verl/verl/trainer/distillation/losses.py#L177`；该副本未分发，见 EXCLUSIONS.json）、task_reward_pg:299（原始源路径 `reports/training_objective_20261023/legacy_4bc0da93/verl/verl/trainer/distillation/losses.py#L299`；该副本未分发，见 EXCLUSIONS.json） | 根据独立权重加入终局奖励的 GRPO clipped policy loss，再可选加入成功专家轨迹 CE |

这一机制符合“在学生访问的轨迹上接受教师监督”的 OPD 判定；OPD 的判定不能仅凭类名中有 PPO/GRPO，或工程目录叫 `trace_opd`。[OPD 方法说明](https://thinkingmachines.ai/blog/on-policy-distillation/)与 [DeepSeekMath §4.1](https://arxiv.org/html/2402.03300v3#S4.SS1)分别说明了教师逐 token 监督与组相对奖励优势的区别。

该早期 forward 实现只累加教师 top-K 项，并做 log-prob 和 loss clamp；没有完整词表求和，也没有 forward tail bucket，因此应称 **top-K forward-KL 蒸馏近似**，不能写成精确完整 KL。后续 compute_reverse_kl_topk:199（原始源路径 `reports/training_objective_20261023/legacy_5beecec5/verl/verl/trainer/distillation/fsdp/losses.py#L199`；该副本未分发，见 EXCLUSIONS.json） 则计算 K+1 桶（top-K 加尾部）的 reverse KL。不能把后续实现倒填到此前所有 run。

## 4 OPD 如何与 GRPO 组合

历史混合目标的代码为：

\[
\mathcal L=w_{\mathrm{OPD}}\mathcal L_{\mathrm{OPD}}+
w_{\mathrm{RL}}\mathcal L_{\mathrm{GRPO}}+
w_{\mathrm{SFT}}\mathcal L_{\mathrm{CE}}.
\]

| 历史臂 | 权重 OPD / RL / SFT | 可核查证据与限制 |
|---|---|---|
| A0 纯 OPD | 1 / 0 / 0 | a800-1p7b-a0-pure.yaml:306（原始源路径 `reports/training_objective_20261023/legacy_4bc0da93/configs/experiments/a800-1p7b-a0-pure.yaml#L306`；该副本未分发，见 EXCLUSIONS.json）；纯蒸馏路径跳过 policy loss，即使入口配置 `adv_estimator=grpo` 也不能称为实际在做 GRPO 更新 |
| A1 OPD+GRPO | 1 / 1 / 0 | a800-1p7b-a1-rl.yaml:284（原始源路径 `reports/training_objective_20261023/legacy_4bc0da93/configs/experiments/a800-1p7b-a1-rl.yaml#L284`；该副本未分发，见 EXCLUSIONS.json）；两项对裸学生轨迹共同优化，在同一次 loss 中相加，不是先 OPD 再 GRPO |
| A2 OPD+SFT | 1 / 0 / 0.5 | 历史 runbook（原始源路径 `reports/training_objective_20261023/legacy_4bc0da93/docs/mixed-objective-runbook.md#L34`；该副本未分发，见 EXCLUSIONS.json） 记录了配置方式和 doc-prefix 变体；本轮没有其独立 launch manifest 与损失曲线 |
| A3 三项混合 | 1 / 1 / 0.5 | 历史 runbook 提供命令与实验计划；不能将可用开关写成已经完成的有效实验 |

早期 RL 的信号是 `won` 二值终局奖励，使用 GRPO 组内相对优势；trainer 默认开启标准差归一化，与后期 classic 关闭 std 的配置不同。`use_task_rewards=False` 也不能据此断言没有 RL：定制的 `rl_loss_coef>0` 分支独立调用 `task_reward_pg`，这正是 A1 的组合位置。

开启 SFT 时，额外专家样本使用独立 `sft_mask`，从 OPD/RL 的 response mask 及其组统计中隔离；doc-prefix SFT 在带文档角色上训练。早期版本还专门对齐 OPD 和 RL 的微批归一化，不能将权重 1:1 解读为梯度范数相等。此结论仅对应恢复的代码版本，不能替代逐 run 的生效配置。

现存七条 `trace_opd_4b_rkl_seqmean_rl` 精选轨迹只是行为样本，名称中的 `rl` 不是系数与 loss 生效的证明。当前仍缺它们对应的原始 launch manifest、代码 revision 和逐步 `distillation/loss`、`rl/pg_loss`、非零优势统计，故不为该 run 追认具体系数或 KL 模式。

## 5 日志可以证明到什么程度

本地 71 个 `outputs/**/main_ppo.log` 全为空。不能用它们确认 ALFWorld、WebShop、SearchQA 的实际损失调用。其余日志中，只有 [classic_ssb3b_pgs/launch.log](https://github.com/thx426/trace-experiment-failure-archive/blob/main/runs/classic_ssb3b_pgs/launch.log#L5012) 保留完整 actor 更新指标：

- step 100 为评测记录，没有完整 actor 更新指标；step 101–167 共 **67 个更新步**有 actor loss、pg loss、正梯度范数和优势权重。
- 67 步 `actor/loss` 与 `actor/pg_loss` 最大绝对差为 **9.90×10⁻¹⁰**；runtime config 的 entropy coefficient 为 0、loss mode 为 vanilla，未出现蒸馏损失指标。这与“该日志片段使用 RL 目标”的实现证据一致。
- `actor/pgs_steered_alpha` 与 manifest 中 **T=176、α₀=1** 的余弦日程吻合，最大绝对差 **5.55×10⁻¹⁷**。奖励、score、won 和 actor 梯度同时存在，能佐证该片段确实执行了 PGS 更新。

这些是该 run 的部分运行证据；不证明其余 run 全程行为，也不单独证明论文报告分数、各样本梯度或因果机制。汇总与逐步源行号见 [objective-inventory.json](https://github.com/thx426/trace-experiment-failure-archive/blob/main/reports/training_objective_20261023/objective-inventory.json) 和 [runtime-metrics.json](https://github.com/thx426/trace-experiment-failure-archive/blob/main/reports/training_objective_20261023/runtime-metrics.json)。

## 6 实验表述修订

| 原表述 | 应使用的表述 |
|---|---|
| “学习规则就是标准 GRPO” | 基线为关闭标准差归一化的 GRPO；机制臂为 `pgs_asym_grpo`，写明 bare-only LOO 与正超额引导优势 |
| 将后期 PGS 表格标作 OPD 或 OPD+GRPO | 改为 PGS / sampling-guided asymmetric GRPO；早期 OPD 放入单独实验组 |
| “教师彻底退出学习目标” | 没有直接教师分布匹配损失；教师仍通过采样分布和分臂规则影响更新 |
| “余弦衰减的是 token 干预力度” | 余弦衰减的是引导臂优势权重，到期停止引导臂 |
| “早期所有训练没有任务奖励” | 仅适用于 RL 权重为 0 的纯 OPD；A1 的实现包含独立 GRPO loss |
| “改成奖励优化后长度漏洞不存在” | 终局奖励不能保证排除冗长；按具体运行报告长度、截断和质量观察 |

训练入口 `scripts/launch_train.py` 的头部说明也已同步更正；只改注释，没有改训练行为。现有中文研究说明已按以上范围修改，并保留 修改前副本（原始源路径 `reports/training_objective_20261023/original_research_note.md`；该副本未分发，见 EXCLUSIONS.json） 与 差异（原始源路径 `reports/training_objective_20261023/research-note.patch`；该副本未分发，见 EXCLUSIONS.json）。当前目录没有英文论文正文，以下两段可作为对应已有方法/实验段落的替换文本，需随正式主表绑定具体运行：

> The classic baselines use outcome-reward GRPO without group-wise standard-deviation normalization. Our PGS configuration instead uses an asymmetric group-relative update: bare trajectories use a leave-one-out baseline computed only from bare trajectories, whereas guided trajectories receive the positive excess reward over the bare-group mean, scaled by a cosine-decayed weight. The actor uses the native clipped policy loss with sequence-mean token-mean aggregation. Document-conditioned teacher scores guide sampling, but no teacher-distribution matching loss is included. Behavior importance weighting is disabled in the archived classic PGS configurations.

> We distinguish these PGS experiments from the earlier on-policy distillation variants. The historical OPD implementation samples bare student trajectories and scores the same response prefixes with a document-conditioned self-teacher under stop-gradient. A top-K distribution-matching loss trains the bare student. The historical A1 configuration adds a separately weighted outcome-reward GRPO loss to this distillation objective; pure OPD sets the RL weight to zero. These historical variants are reported separately from PGS, and individual results require matching launch configurations and runtime loss records.

## 7 复核与剩余证据

复核命令：`python3 <PROJECT_ROOT>/reports/training_objective_20261023/recheck.py`。脚本核验历史 Git blob、当前源码快照、22 份启动参数及 67 步日志；不启动训练，不改历史配置与日志。

本轮已解决实现身份、直接训练信号、OPD+GRPO 组合位置和现有表述的对应关系。仍需实验负责人补充早期 run 的 manifest、代码版本、原始损失曲线和后期非 SSB 的训练日志，才能完成逐运行的运行证明。论文 R12/R17/R18/R22/R23/R34 的全面门槛不会因这一项技术核对自动通过；不将本轮审计称为整篇论文投稿就绪。
