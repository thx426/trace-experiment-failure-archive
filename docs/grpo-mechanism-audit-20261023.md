# 8 月实验的 Skill、奖励与 GRPO 机制核对

核对日期：2026-10-23。**历史实验仅指项目 `runs/classic_*` 中 2026 年 8 月的实验。10 月 API 试跑不属于本报告的历史证据。** 用户已说明 GPU 由另一线程部署，本轮停止云端操作，只进行本地文件核对、CPU 算法对照及候选开关测试，没有启动 GPU 训练。

结论：当前实现与“Skill 改变采样、奖励驱动学生更新”的设计相符，但它是**不对称分臂 GRPO 变体**，不能等同标准 on-policy GRPO。现有记录支持部分任务、部分阶段的收益，尚不支持“Skill 普遍改善最终训练效果”。优先处理评测口径混杂，再验证缺失 Skill 的分组处理和奖励稀疏性；不能把算法对照通过当作训练效果已提升。

## 1. 已核实的历史证据

本地有 11 个 run、22 份有效的 8 月启动 manifest，另有 1 份零字节 manifest。各有效 manifest 的优势估计器与实际命令一致。研究员配置指向内网 Qwen3.6-35B-A3B 服务；项目还保存了自部署启动脚本。字段 `api_key_env` 是接口认证配置，不能据此把它判定为商业 API 实验。[WebShop PGS 启动记录](https://github.com/thx426/trace-experiment-failure-archive/blob/main/runs/classic_ws3b_pgs/launch_manifest_20260817_002909.json)

当前 minimal 文件夹中，这 11 个 run 的原始 `rollouts/*.json`、研究员 `doc.md` 和 `episodes.jsonl` 数量均为 0；71 个 8 月 `main_ppo.log` 为空。保留了评测汇总 `probe/val_scan_out.csv` 和部分臂统计，不能由此还原每条 Skill 内容、信息来源、奖励和更新。没有记录源码 commit 的历史 manifest，也不能证明当前代码与当时逐字节相同。

SSB 保留了完整日志片段：step 101–167 共 67 次更新，actor loss 与 policy-gradient loss 最大差为 **9.90×10⁻¹⁰**；引导优势权重与 manifest 的 T=176 余弦日程最大差为 **5.55×10⁻¹⁷**；日志中平均 reward 与 won 相同。这能证实该片段的更新和日程，不能替代其他 run 的执行证据。[SSB 逐步复核](https://github.com/thx426/trace-experiment-failure-archive/blob/main/reports/grpo_mechanism_20261023/august/ssb-runtime.json)

所有原始输入均记录 SHA256；重新计算脚本不访问网络、不调服务、不占 GPU。[证据清单](https://github.com/thx426/trace-experiment-failure-archive/blob/main/reports/grpo_mechanism_20261023/august/source-manifest.json) · [机器可读核对结果](https://github.com/thx426/trace-experiment-failure-archive/blob/main/reports/grpo_mechanism_20261023/august/audit.json)

## 2. 算法机制是否匹配

当前链路是：研究员在环境中做实验 → 生成 Skill → 带 Skill 的上下文给学生 token 打指导分 → SPS 改变部分学生轨迹的采样分布 → 终局奖励 → 分臂优势 → 学生在无 Skill 的训练上下文上执行 clipped policy-gradient 更新。它不是把研究员完整轨迹当作 SFT 数据，也没有后期 classic PGS 的 OPD 分布匹配损失。早期 OPD 版本另见[训练目标核对](https://github.com/thx426/trace-experiment-failure-archive/blob/main/docs/training-objective-audit-20261023.md)。

令同任务裸臂为 B、引导臂为 S，终局奖励为 R，正常完整组的实际优势为：

\[
A_i^B=R_i-\frac{1}{|B|-1}\sum_{j\in B,j\ne i}R_j,
\qquad
A_i^S=\alpha_t\max(R_i-\overline R_B,0).
\]

两臂都不做组内标准差归一化。历史 GRPO 基线则用整个组的 `R−mean(R)`。因此 PGS 同时改变采样分布、基线和引导臂正负梯度规则，不能把组间差异全归因于 Skill。[生产优势函数](verl/verl/trainer/ppo/core_algos.py#L721)

当前实现还有四个关键边界：

- `w_beh_mode=off`，PPO 的 `πθ/πold` 不等于对 Skill 行为分布 b 的完整修正；不宣称无偏 on-policy 更新。GRPO 的一般机制参见 [DeepSeekMath §4.1](https://arxiv.org/abs/2402.03300)。
- 终局优势广播到 response mask 中的全部学生生成 token；环境观察 token 不参与损失。该 mask 没有验证每个生成 token 使用的信息是否可由裸学生获得。
- `αt=(1+cos(πt/T))/2` 衰减的是引导臂优势，到 T 才停止引导与研究。它不是直接退火 token 干预幅度。WebShop T=133；60 步时 α≈0.576，60 步实验不能验证退出 Skill 后的效果。
- 源码中部分早期注释仍描述“混合组基线惩罚裸失败”；当前 `pgs_asym_grpo` 使用 bare-only LOO，裸臂全失败时裸臂优势全为零，应以实际函数和生效配置为准。

## 3. 会影响效果或结论的环节

| 环节 | 证据与影响 | 优化及验证方式 |
|---|---|---|
| **SearchQA 评测混杂** | 8 月中旬从 greedy@1 切到 temperature=0.6、n=3；月底 PGS manifest 改为 `val_eq320`，汇总仍含 2048 任务。step 50 有 10240 条/2048 任务，即平均 5 条，不是干净的 avg@3；旧脚本也记录了同目录追加两种口径的问题。 | 按评测 run ID、时间、任务清单、checkpoint、采样参数分区，拒绝重复副本和混合轮次；重新做相同解码、相同任务的无 Skill 评测。这些 SQA 汇总不进入当前同口径比较。 |
| **Skill 不等于可迁移证据** | 当前 `_check_doc` 检查词数、关键词、动作拷贝和零成功宣称；首次警告后原样再交也会接受。它不是事实核验或裸学生可获信息核验。缺少 8 月原始文档，无法量化当时发生比例。 | 保留“策略”和“本任务具体答案/定位符/属性”来源；用独立验证轨迹和裸学生可见观察核验。按任务做完整 Skill/策略化 Skill 的配对实验；现有定位符屏蔽只是保守消融，不能视为语义修复完成。 |
| **二值奖励与分臂稀疏性** | 当前默认 `reward_from_score=false`；历史 manifest 未序列化该开关，应将其标作未记录，不能仅靠默认值追认所有历史 run。若使用二值奖励，部分完成没有奖励差异；裸臂全失败则无裸臂学习信号，引导失败也为零。 | 显式保存 reward 与 score，统计全零优势组。比较 binary/score 时同时评估完整成功率；连续分数改变了优化目标，不保证提高成功率。 |
| **缺失 Skill 的优势归属** | 当前默认保留分配臂：文档缺失/超长后用裸采样，但仍可能按引导臂只保留正优势。算法上可确认，8 月发生率未知。 | 候选 `effective_bare` 只对生成前已确定的缺文档/溢出/屏蔽回退计算裸臂优势；另保留原 `steer_arm` 作实验分组。不能按失败、零干预次数或拒绝数事后挑样本。 |
| **更新尺度及策略偏差** | 裸臂 LOO 相比相同裸组含自身均值的优势放大 m/(m−1)：m=4 为 4/3，m=8 为 8/7。两者还使用不同大小/分布的组。长轨迹的重要性权重存在偏差与方差折中。 | 增加无 Skill 的 GRPO/LOO 对照；行为权重需另做推导和方差检查。不能简单打开 per-token 权重就宣称无偏。 |
| **预算和退场阶段** | 研究员每任务可用多轮额外实验；相同步数不等于相同总计算或环境交互预算。ALF 的两臂 task timeout 还不同。 | 同时报告学生步数、总环境调用、研究员 token、GPU 时、墙钟和超时率；预注册固定末步与学习曲线，覆盖 T 后裸策略阶段。 |

[Skill 校验实现](src/trace_opd/research/tools.py#L364) · [奖励取数](src/trace_opd/train/agent_loop.py#L1053) · [SearchQA 旧评测警示](https://github.com/thx426/trace-experiment-failure-archive/blob/main/probe/val_avg3.py#L1)

## 4. 8 月已有对照结果能说明什么

以下直接用原汇总的 `n_won/n_traj` 重算，比较同一步数。它们是历史汇总描述，缺少原始逐任务配对记录和多训练种子，**不报告显著性或因果归因**；不能用当前最新 manifest 替每个旧 checkpoint 补齐来源。

| 任务 / 同步检查点 | PGS 成功率 | GRPO 成功率 | 差值 |
|---|---:|---:|---:|
| WebShop 3B / step 20 | 46.27% | 33.07% | +13.20 pp |
| WebShop 3B / step 60 | 70.27% | 72.53% | −2.27 pp |
| WebShop 3B / step 200 | 80.73% | 83.60% | −2.87 pp |
| ALFWorld 3B unseen / step 20 | 40.80% | 51.99% | −11.19 pp |
| ALFWorld 3B unseen / step 140 | 96.27% | 88.81% | +7.46 pp |
| ALFWorld 7B unseen / step 180 | 98.51% | 97.01% | +1.49 pp |

WebShop 支持“初期学习更快”，不支持“最终更好”；ALFWorld 3B 在现存较晚共同检查点上更好。ALF 两模型的引导退出边界 T=300 尚未被上述 PGS 曲线覆盖，不能声称已验证教师退场后的持久收益。SSB/Math 在本地汇总中没有对应 GRPO 曲线。`arm_series_all.json` 的 WebShop 指标是连续 **score**，不是本表成功率；也不是独立裸策略评测。[全部共同检查点](https://github.com/thx426/trace-experiment-failure-archive/blob/main/reports/grpo_mechanism_20261023/august/matched-checkpoints.csv)

## 5. 本轮完成的 CPU 对照

执行当前生产优势函数，组大小取 8 月配置中的 n=8、16；固定合成奖励，不调用模型、不做优化器更新。

| 固定条件 | 原机制 | 对照结果 | 可以得出的结论 |
|---|---|---|---|
| 裸臂全失败、引导臂全成功，α=1 | 裸臂 A=0，引导 A=1 | 全组均值基线：裸 A=−0.5，引导 A=0.5 | 两个目标确有不同的负向学习信号；不是可互换的实现 |
| 文档全部缺失，两臂各半成功 | 原引导失败 A=0 | 全部有效裸臂：n=8 失败 A≈−0.5714；n=16 失败 A≈−0.5333 | 回退分组能恢复失败信号；不证明最终成功率会提升 |
| 裸奖励均为 0，引导部分分数为 0.4/0.8 | binary 下全部 A=0 | score 下引导 A=0.4/0.8 | 连续奖励增加信号，同时改变偏好目标 |
| 两步决策，π=0.2、b=0.6，终奖为两步均成功 | 真策略梯度 [0.032,0.032] | 无校正 [0.288,0.288]；逐 token [0.096,0.096]；前缀乘积 [0.096,0.032] | 对终局回报，逐 token 或仅前缀校正一般不充分；全轨迹校正在该小例中精确，长轨迹方差另需验证 |

最后一项由完整枚举得到，使用生产 `expand_behavior_weights`，无抽样误差，但不代表真实模型的梯度幅度。[合成对照结果](https://github.com/thx426/trace-experiment-failure-archive/blob/main/reports/grpo_mechanism_20261023/august/audit.json)

**57 项本地测试通过**，覆盖回退分组、保留实验臂标签、padding、孤立引导组、mask、T 边界、旧默认指纹保持、候选配置指纹区分和现有 task gate 兼容性。新开关默认 `assigned`；未启用就保留原更新规则。[测试记录](https://github.com/thx426/trace-experiment-failure-archive/blob/main/reports/grpo_mechanism_20261023/tests.txt) · [候选分组实现](src/trace_opd/steering/arms.py#L28)

## 6. 待执行的训练效果验证

准备了从 8 月 WebShop 配置出发的方案，**没有启动，也没有向另一线程发送指令**。研究员部署和 GPU 使用由用户指定的另一线程负责。

| 对照 | 唯一主要变化 | 用途 |
|---|---|---|
| P0 ↔ P1，优先 | `fallback_advantage=assigned` ↔ `effective_bare` | 检验回退分组是否改善无 Skill 成功率；若没有回退样本，不能检验该机制 |
| G0 ↔ G1 | 无 Skill，`grpo` ↔ `grpo_loo` | 测量基线定义和尺度影响 |
| P0 ↔ R1，可选 | binary ↔ continuous score | 评估完整成功率与部分分数的取舍 |

各配对须固定同一开源研究员权重、服务版本、学生模型、数据顺序、GPU 配置、rollout 数和解码参数；不能直接用近期 API 结果作对照。先用 seed 42 检查机制与运行，候选有效后增加独立训练种子。WebShop 完整比较至 step 200，保留 step 140–200 的退出引导后评测；终评不带 Skill，固定 500 个任务×3 次采样。调参使用另留的 dev 集，不挑最佳 test checkpoint。200 步及资源分配是实验方案，须由占用 GPU 的线程统一安排。

精确参数、所需证据和未执行状态均记录在[实验方案](https://github.com/thx426/trace-experiment-failure-archive/blob/main/reports/grpo_mechanism_20261023/august/experiment-plan.json)。目前完成的是历史配置/汇总核对、机制问题验证和候选实现；**优化后的模型训练收益尚未验证**。

## 7. 复核命令

在项目根目录执行以下命令只使用本地 CPU：

```bash
PYTHONPATH=src uv run --python 3.13 --with numpy --with torch \
  python scripts/classic/audit_august_grpo.py \
  --output reports/grpo_mechanism_20261023/august

PYTHONPATH=src uv run --python 3.13 --with numpy --with pyyaml \
  --with pytest --with torch --with httpx python -m pytest \
  tests/test_grpo_mechanism_replay.py tests/test_fallback_advantage.py \
  tests/test_training_audit.py tests/test_information_dependencies.py \
  tests/test_pass5_gate.py -q
```

此前 10 月试跑分析仍保留，已在[范围更正说明](https://github.com/thx426/trace-experiment-failure-archive/blob/main/reports/grpo_mechanism_20261023/README.md)明确排除；其轨迹数量、缺文档比例和配对差值不用于上述 8 月结论。
