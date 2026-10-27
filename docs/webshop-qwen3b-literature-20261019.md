Qwen2.5-3B-Instruct 的 WebShop 改进证据与实验候选

检索日期：2026-10-19。本文用于实验设计，不是新实验结果。本轮没有修改训练代码或重启云端任务。

目标是提高撤掉研究员和指导文档后的学生完全成功率。当前骨干是 **Qwen2.5-3B-Instruct**。本文中的“3B”均指参数规模。它不是 Qwen3 的型号名称。

先给出判断：最值得测试的是动作级信用分配、成功轨迹课程、短成功示范训练，以及更准确的指导位置。先修复文档交付，再完成现有比例对照。每次只增加一个科学变量。没有文献能保证某项改动会使本项目的成功率达到指定数值。

检索覆盖 ACL 主会及 Findings ACL、ICML、AAAI、NeurIPS 的相关工作，重点为 2022–2026 年截至检索日已公开的论文。检索使用会议官方站点、论文标题、WebShop、Qwen、agent training、credit assignment、entropy、SFT、replay、distillation、search 等关键词，并沿相关文献引用补查。正式归属由 ACL Anthology、PMLR、AAAI OJS 或 NeurIPS proceedings 核验。这不是可证明穷尽的系统综述。不能写成“已阅读这些会议的所有论文”。

本轮核心清单共34篇，其中32篇属于上述会议（含Findings），2篇为单列预印本。下方逐篇表区分四种证据：同模型同任务；同任务不同模型；其他任务的迁移线索；增加推理辅助。Findings 单独标明。另列两篇同模型预印本。AutoGuide 的完整 PDF 因工具大小限制未能打开，其表格来自官方 PDF 的检索索引；其余条目的关键方法或结果表已核对。

当前记录给出了三个实验线索，不能把它们当作根因证明。

| 已核对事实 | 可能说明什么 | 尚不能证明什么 |
| --- | --- | --- |
| 第 14 次更新使用的训练批次，两臂均为 74/256 次成功，即 28.90625% | 同批引导没有提高完全成功次数 | 不能推断整个训练无效，也不是更新后的开发集成绩 |
| 自主平均 score 为 55.23；引导为 61.39 | 部分失败更接近目标，值得检查动作学习信号 | 不能把多出的 6.16 分算成成功率提升 |
| 引导臂 64/256 条没有文档，实际未引导 | 名义比例和实际干预比例不同 | 不能只用这批数据测量完整文档方法的效果 |
| 引导臂 182 条失败均为零优势 | 当前目标没有直接从这些失败获得引导臂策略梯度 | 不能直接断言正优势筛选错误；失败文本也可能含有害动作 |
| 该批截断和格式错误为零 | 暂无证据把截断或动作语法列为首要问题 | 不能排除合法但无效动作、提前购买或选项理解错误 |

本地依据：[运行记录](https://github.com/thx426/trace-experiment-failure-archive/blob/main/docs/webshop-arm-ratio-runbook.md#L19)。本轮没有重新连接云端，状态只引用该记录的检查时间。当前配置已关闭优势标准差归一化，并已限制输出长度；这些不能再次写成新改进。

下面的候选表将论文中的方法与本项目改动分开。P1 表示优先做独立小实验；P2 表示有诊断支持后再测；P3 表示较大改动或主要增加推理成本。所有“本项目测试”都是迁移提案，不是原论文已验证的 Qwen2.5-3B 结果。

| 编号 | 候选改动与依据 | 本项目的最小测试及限制 | 优先级 |
| --- | --- | --- | --- |
| 01 | 动作级相对优势：[GiGPO](https://papers.neurips.cc/paper_files/paper/2025/hash/420c9f777c0b4f78d515e53cf74d58b2-Abstract-Conference.html) | 先测同状态分组覆盖率。只改变优势估计，保持终奖、采样和裸评测。 | P1 |
| 02 | 动态感知状态分组：[SHADOW](https://ojs.aaai.org/index.php/AAAI/article/view/39570) | 检查相同观察是否隐藏不同已选选项或后继状态。有碰撞时再升级分组。 | P2 |
| 03 | 状态归一化熵：[STAPO](https://aclanthology.org/2026.acl-long.1308/) | 先验证熵能否识别重要动作。不要把该文的完整优化收益当作换门控阈值的收益。 | P1诊断/P2训练 |
| 04 | 按不确定性与结果调整梯度：[EMPG](https://proceedings.mlr.press/v306/wang26ek.html) | 与高熵加权分开比较。低熵动作也可能是稳定的正确动作，或自信错误。 | P2 |
| 05 | 子目标与动作分层学习：[HiPER](https://proceedings.mlr.press/v306/peng26h.html) | 仅在跨商品、选项和购买子目标切换错误明显时考虑。需要新输出结构和 critic。 | P3 |
| 06 | 成功轨迹后缀课程：[CRPS](https://aclanthology.org/2026.findings-acl.680/) | 从可恢复的训练中间状态开始，逐渐增加学生负责的后缀；保留从起点探索。测试仍从起点开始。 | P1 |
| 07 | 短成功轨迹 SFT/RFT：[T²PO](https://proceedings.mlr.press/v306/wang26ah.html)、[AgentTuning](https://aclanthology.org/2024.findings-acl.181/) | 使用训练侧验真轨迹。与无 SFT 组使用同样后续 RL；另计示范生成和 SFT 成本。 | P1 |
| 08 | 成功轨迹迭代自训练：[AgentGym](https://aclanthology.org/2025.acl-long.1355/) | 保存新成功案例，做独立自训练分支。不能把额外探索视为免费。 | P2 |
| 09 | 成功/失败轨迹偏好对：[ETO](https://aclanthology.org/2024.acl-long.409/) | 对同任务构造有效偏好；单独测试 DPO 类目标，避免与比例实验混合。 | P2 |
| 10 | 关键步骤 SFT：[ATLAS](https://aclanthology.org/2025.findings-acl.1299/) | 同数据对照完整轨迹训练。Qwen7B 原文已有负结果，不预设更少步骤更好。 | P2 |
| 11 | 学习真实动作后果：[Early Experience / IWM](https://proceedings.mlr.press/v306/zhang26m.html) | 针对合法无效动作，构造少量 state-action-next-state 监督；所有后果来自真实环境。 | P2 |
| 12 | 基于实际后果的反思监督：[Early Experience / SR](https://proceedings.mlr.press/v306/zhang26m.html) | 比较备选动作的真实后继状态。不能仅让研究员写更长的主观解释。 | P2 |
| 13 | 提示和 SFT 数据能力配比：[Agent-FLAN](https://aclanthology.org/2024.findings-acl.557/) | 只有数据格式或能力配比存在问题时采用；区分格式能力与选商品能力。 | P2 |
| 14 | 状态相关的短指南：[AutoGuide](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d8efbb5dd415974eb095c3f06bff1f48-Abstract-Conference.html)、[WKM](https://papers.neurips.cc/paper_files/paper/2024/file/d032263772946dd5026e7f3cd22bce5b-Paper-Conference.pdf) | 在相同文档 token 预算下，比较整份 Skill 和当前状态适用片段。测试仍撤掉指南。 | P1 |
| 15 | 成功与失败配对提炼规则：[ExpeL](https://ojs.aaai.org/index.php/AAAI/article/view/29936)、[WKM](https://papers.neurips.cc/paper_files/paper/2024/file/d032263772946dd5026e7f3cd22bce5b-Paper-Conference.pdf) | 固定研究员、任务和调用预算。让规则包含可观察条件、动作及验证依据。 | P1 |
| 16 | 用教师判断定位干预：[UOPD](https://arxiv.org/html/2609.34036v1)（预印本） | 比较学生熵与教师对学生动作的置信度；加入相同实际干预次数的随机对照。 | P2 |
| 17 | 把纠正动作变成显式监督：[UOPD](https://arxiv.org/html/2609.34036v1)、[SEED](https://arxiv.org/html/2607.14777v1)（预印本） | 独立测试 RL 加辅助监督。不能把改变采样后自然变好的轨迹等同于成功内化。 | P2 |
| 18 | 平滑减少辅助：[UFT](https://proceedings.neurips.cc/paper_files/paper/2025/file/92e886487a8354b03d8bf4416eae6d7d-Paper-Conference.pdf)、[UOPD](https://arxiv.org/html/2609.34036v1) | 比较采样比例、实际动作干预率和梯度权重三种不同的衰减。它们不是同一开关。 | P2 |
| 19 | 混合外部成功轨迹：[LUFFY](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a9d5c33e38442450f9781bd7bad3c81e-Abstract-Conference.html) | 使用适合混合行为策略的目标。不能直接把教师或旧轨迹当作当前 on-policy 样本。 | P2 |
| 20 | 筛选有结果差异的任务组：[DAPO](https://papers.nips.cc/paper_files/paper/2025/file/a4277440d50f1f15d2cb4c14f7e0c0d2-Paper-Conference.pdf) | 先统计全零/全一组。若重采样，记录所有被放弃的轨迹与研究员成本；保留组内失败样本。 | P2 |
| 21 | 提高有效动作多样性：[ActRe](https://ojs.aaai.org/index.php/AAAI/article/view/40794)、WebShop原文 | 分别测思考熵、动作熵、不同搜索词、不同商品和有效选项。先局部改动作探索。 | P1诊断/P2训练 |
| 22 | 语义多样性重加权：[MMR-GRPO](https://aclanthology.org/2026.findings-acl.467/) | 先判断重复是否严重。数学1.5B未显示加速，不能保证3B agent有效。 | P2 |
| 23 | PPO 非对称 clipping：[DAPO](https://papers.nips.cc/paper_files/paper/2025/file/a4277440d50f1f15d2cb4c14f7e0c0d2-Paper-Conference.pdf) | 仅在熵坍缩和 clipping 数据支持时测。它不是当前 steering.epsilon=0.05。 | P2 |
| 24 | token 级损失聚合：[DAPO](https://papers.nips.cc/paper_files/paper/2025/file/a4277440d50f1f15d2cb4c14f7e0c0d2-Paper-Conference.pdf) | 与现有 seq-mean-token-mean 单独比较，并按轨迹长度分层。不能修复本来为零的优势。 | P2 |
| 25 | 高熵 token 梯度 mask：[Beyond 80/20](https://papers.nips.cc/paper_files/paper/2025/hash/a797c2d2e0c1fdabf4d1ab8cd0b465c6-Abstract-Conference.html) | 仅有更大模型数学证据。不能拿它证明文档熵门控有效。 | P3 |
| 26 | 连续分数或局部进度信号：WebShop原文及本地记录 | 单独更改训练信号；完全成功率仍为主指标，评分器保持不变。警惕满足部分要求后提前购买。 | P1/P2 |
| 27 | 成功约束下优化长度：[DEPO](https://ojs.aaai.org/index.php/AAAI/article/view/40279) | 先建立成功能力，再偏好成功且短的路径。失败和成功的长度分别报告。 | P2 |
| 28 | 动态思考长度和停滞重采样：[T²PO](https://proceedings.mlr.press/v306/wang26ah.html) | 只有思考冗长/停滞确实存在才测。Qwen3的思考结束标记不能照搬到Qwen2.5。 | P2 |
| 29 | 真正超长样本的处理：[DAPO](https://papers.nips.cc/paper_files/paper/2025/file/a4277440d50f1f15d2cb4c14f7e0c0d2-Paper-Conference.pdf) | 区分时间超时、上下文截断、动作数上限。当前第14批截断为零，优先级低。 | P3 |
| 30 | 历史压缩与关键信息保留：WebShop原文 | 单测目标、价格、选项等信息是否被裁掉。更多历史不是必然更好；保持任务起点和评分一致。 | P2 |
| 31 | Q值/奖励模型动作重排：[AgentRM](https://aclanthology.org/2025.acl-long.945/)、[QLASS](https://proceedings.mlr.press/v267/lin25l.html)、[Step-Q](https://ojs.aaai.org/index.php/AAAI/article/view/34924)、[LAC](https://proceedings.mlr.press/v267/dong25c.html) | 先测候选中有无正确动作；可作为训练数据生成器。推理时使用则单列模型、候选数和token成本。 | P3 |
| 32 | 树搜索或多次候选选择：[LATS](https://proceedings.mlr.press/v235/zhou24r.html)、[SoftSC](https://aclanthology.org/2024.acl-short.28/) | 用于诊断能力上界或生成示范。搜索后挑最好结果不是裸学生 avg@3。 | P3 |
| 33 | 重试反思记忆：[Reflexion](https://papers.neurips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html) | 只作有成本的探索性对照。原论文WebShop未改善，且不更新模型参数。 | P3 |
| 34 | off-policy critic与回放：[ArCHer](https://proceedings.mlr.press/v235/zhou24t.html) | 仅在数据复用收益能抵消critic复杂度时考虑。不能把旧轨迹直接重放进普通GRPO。 | P3 |

动作级信用分配也有边界：如果相关分组的所有回报都相同，它仍可能没有有效对比信号。此时需要成功探索、课程或经过验证的额外监督，而不是只把零优势拆到更多步骤。

这些候选不能一次全部加入。下面给出可执行顺序。

1. 先恢复可信的指导条件。区分 API 拒绝、文档生成失败、空文档和未触发引导。将名义引导数、有效文档数、实际改动作数分别记录。缺文档样本不能静默当作“获得引导”的机制证据。若改变运行协议，两个实验臂应采用同一协议并使用新 run ID。
2. 完成原定 8自主+8引导 对 12自主+4引导。继续固定 T=133、steering epsilon=0.05、二值奖励和其余科学配置。修复运行问题不等于取得算法增益。第14批只用于诊断。
3. 建立同期裸 GRPO 基线，再做一个动作级优势分支。优先 GiGPO 风格的无 critic 方案。先在自主样本上验证；引导样本的行为策略和非对称优势是额外变量，不直接与新估计器混合。若同状态覆盖不足，先解决分组问题。
4. 单独做短 SFT 对照。首先用少量训练侧已验真的成功轨迹，覆盖搜索、比较、选项选择和购买前核验。以一轮训练作为待检验起点，不能称为文献确定的最优值。比较 SFT 后与无 SFT 的 dev0，再给相同 RL 预算。SFT 本身也可能退化。
5. 依据诊断选择第三条分支。全失败组多时测试 CRPS；局部指南不相关时测试状态相关文档；部分完成长期无法变成成功时测试连续训练信号。一次只选一条，不做三个改动的组合。

如果计算资源有限，先运行第2项，接着运行第3、4项。第5项根据已有轨迹决定。完整 HiPER、MCTS、额外奖励模型和复杂 world model 放到后续。它们更难隔离收益来源，也更偏离当前论文的最小方法。

应采用统一的判定方式：开发集继续固定为256题，每题3次，温度0.6。分别报告完全成功率、平均连续分数、按任务配对的差值区间、训练/研究员/环境的实际成本。平均3次成功率与至少成功一次的 pass@3 分开报告。方案确定后才使用最终500题测试集。筛选阶段可先用一个种子；保留的方案需要多次独立训练。多个评测种子不能代替多个训练种子。

统计重复和离群值时要保留科学含义。相同 sample_id/attempt_id 的重复落盘可去重；独立采样得到相同行为应计入多样性指标，不能从成功率分母删掉。异常高低分先检查记录来源。真实的困难样本、低分失败和罕见成功都不是应自动删除的坏数据。STAPO使用离群信息分配学习权重，并非支持删除不利结果。

论文写作可以采用以下表述：“现有实验显示，文档引导提高了部分轨迹的连续完成分数，但尚未增加该训练批次的完全成功次数。文献支持进一步检验步骤级学习信号、训练课程和成功示范初始化。上述结论为实验假设，不代表已证实的性能改进。”

逐篇证据表如下。SR表示完全成功率；score表示连续完成分数。表中每个差值只属于对应论文的实验条件，不能跨行排列名次。

| 文献与正式来源 | 模型／任务 | 核对到的证据 | 限制与位置 |
| --- | --- | --- | --- |
| [STAPO](https://aclanthology.org/2026.acl-long.1308/)；ACL 2026 主会长文 | Qwen2.5-1.5B/7B；WebShop | 1.5B：GRPO SR56.8%，完整STAPO69.0%；同状态熵归一化与选择性信用分配。 | 表1/3，PDF第6–7页。完整方法不是单独换熵阈值；3B结果是QA。 |
| [CRPS](https://aclanthology.org/2026.findings-acl.680/)；Findings ACL 2026 | Qwen2.5-1.5B/7B；WebShop | 成功后缀课程：1.5B SR53%→61%；7B 68%→74%。 | 表1，第7页。需要中间状态恢复；无3B直接证据，表中无误差条。 |
| [ATLAS](https://aclanthology.org/2025.findings-acl.1299/)；Findings ACL 2025 | Llama3.1-8B、Qwen2.5-7B；WebShop | 关键步骤SFT：Llama SR68→71.5；Qwen7B却由65.5降至62.5。 | 表1/2及附录表7。不能宣称关键步骤SFT普遍优于全轨迹。 |
| [AgentTuning](https://aclanthology.org/2024.findings-acl.181/)；Findings ACL 2024 | Llama2-chat；WebShop等 | 筛选成功轨迹并混通用指令；7B平均reward由4.4到63.6。 | 表3及表5。63.6不是成功率；旧框架版本影响复现。 |
| [AgentGym / AgentSTaR](https://aclanthology.org/2025.acl-long.1355/)；ACL 2025 主会长文 | Llama2-Chat-7B；WebShop等 | 成功探索后迭代训练：初始SFT SR66.5，大数据SFT73.5，自训练76.5。 | 表3，第6页。包含额外探索和训练；不是GRPO。 |
| [Agent-FLAN](https://aclanthology.org/2024.findings-acl.557/)；Findings ACL 2024 | Llama2；WebShop包含在多任务训练中 | 对齐聊天格式、能力配比及负样本；可用于设计SFT数据。 | 表2/6。没有单独WebShop增益，不能搬用跨任务均值。 |
| [AgentRM](https://aclanthology.org/2025.acl-long.945/)；ACL 2025 主会长文 | Llama3-8B策略＋8B RM；WebShop | 任务SFT策略平均reward：greedy61.4，Best-of-5为71.0，beam为75.3。 | 表2/8。需要推理RM及搜索；数值不是SR。 |
| [SoftSC](https://aclanthology.org/2024.acl-short.28/)；ACL 2024 主会短文 | CodeLlama-34B；WebShop | 按token概率重排候选，SR greedy16.0→SoftSC(k10)24.6。 | 表1，50题。是推理多采样收益，不是小模型参数学习。 |
| [MMR-GRPO](https://aclanthology.org/2026.findings-acl.467/)；Findings ACL 2026 | 蒸馏Qwen1.5B/7B、Llama8B；数学 | 按语义多样性重加权；7B更早达峰，1.5B没有加速。 | 表1，第7页。无WebShop；不是删除重复落盘记录。 |
| [ETO](https://aclanthology.org/2024.acl-long.409/)；ACL 2024 主会长文 | Llama2-7B-Chat；WebShop | SFT后做探索偏好训练：SR33.0→37.5；reward63.1→67.4。 | 表2、附录表7。先SFT3epoch，再两轮探索/DPO；无SFT直接ETO会退化。 |
| [T²PO](https://proceedings.mlr.press/v306/wang26ah.html)；ICML main conference 2026 | Qwen3-4B/8B；WebShop | 4B完整SR81.64；去RFT61.32；控制思考长度与停滞重采样。 | 表1/3/4。非Qwen2.5；评测是否关闭控制器未核明。 |
| [Early Experience](https://proceedings.mlr.press/v306/zhang26m.html)；ICML main conference 2026 | Qwen2.5-7B等；WebShop | 7B：raw GRPO SR64.8；IL+GRPO84.4；IWM+GRPO91.4。 | 表5，第20页。额外1571人类轨迹和真实后继数据；不能当等总成本。 |
| [HiPER](https://proceedings.mlr.press/v306/peng26h.html)；ICML main conference 2026 | Qwen2.5-1.5B/7B；WebShop | 7B GiGPO SR75.2→HiPER83.3；对子目标和动作分层分配优势。 | 表1，第9页。增加critic和输出协议；部分基线外引。 |
| [QLASS](https://proceedings.mlr.press/v267/lin25l.html)；ICML main conference 2025 | Llama2-7B；WebShop | Q引导搜索score70.3；将搜索结果用于自训练时，score63.1→66.4。 | 表2及图4。200题、5步；主表依赖搜索和额外QNet，不是SR。 |
| [LATS](https://proceedings.mlr.press/v235/zhou24r.html)；ICML main conference 2024 | GPT-3.5；WebShop | 树搜索score75.9、SR38.0；两者不可互换。 | 表6，第8页。免参数更新，增加分支、估值和反思调用。 |
| [LAC / Actor-Critic](https://proceedings.mlr.press/v267/dong25c.html)；ICML main conference 2025 | CodeLlama/Gemma/Llama/Mistral；WebShop | 候选动作的GOOD/BAD概率和未来预测用于推理重加权。 | 附录表3/4。无3B；多候选和世界预测有额外成本。 |
| [ArCHer](https://proceedings.mlr.press/v235/zhou24t.html)；ICML main conference 2024 | GPT2策略＋RoBERTa critic；WebShop | 动作级off-policy价值与token级策略学习结合，使用经验回放。 | 图4及附录A/C。100题快速子集、10步；没有可直接引用的最终SR。 |
| [EMPG](https://proceedings.mlr.press/v306/wang26ek.html)；ICML main conference 2026 | Qwen2.5-1.5B/7B；WebShop | 7B GRPO SR65.6→69.3；用动作熵与终奖调整梯度。 | 表1，第7页。低熵动作可获得更强正负更新；组件消融非WebShop。 |
| [WebShop原论文](https://proceedings.neurips.cc/paper_files/paper/2022/hash/82ad13ec01f9fe44c01cb91814fd7b8c-Abstract-Conference.html)；NeurIPS 2022 Main Conference | BART搜索＋BERT动作选择 | IL→IL+RL：score59.9→62.4，但SR29.1→28.7。 | 图4，第8页。直接反例：连续分数上升不保证完全成功率上升。 |
| [Reflexion](https://papers.neurips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html)；NeurIPS 2023 Main Conference | ReAct大模型代理；WebShop | 100个环境、至多4次尝试，原作者未观察到WebShop改善。 | 附录B.1，第14页。反思记忆不更新参数；应保留负面证据。 |
| [ExpeL](https://ojs.aaai.org/index.php/AAAI/article/view/29936)；AAAI 2024, 38(17), 19632–19642 | GPT-3.5；WebShop | 从成功/失败提炼规则并检索案例；研究规则与案例的不同作用。 | 第5.2节。推理读取经验；迁移为训练文档后仍需验证裸模型。 |
| [AutoGuide](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d8efbb5dd415974eb095c3f06bff1f48-Abstract-Conference.html)；NeurIPS 2024 Main Conference | GPT-3.5；WebShop | 状态相关指南：ReAct SR30→46；同文另有更多示例对照。 | 表1/3仅核官方PDF索引。完整PDF未打开；原方法推理使用指南。 |
| [WKM](https://papers.neurips.cc/paper_files/paper/2024/file/d032263772946dd5026e7f3cd22bce5b-Paper-Conference.pdf)；NeurIPS 2024 Main Conference | Mistral7B等；WebShop | 任务知识与状态知识分别帮助决策；完整方法Mistral score65.48。 | 表1及图3。额外知识模型与检索；不是65.48% SR。 |
| [GiGPO](https://papers.neurips.cc/paper_files/paper/2025/hash/420c9f777c0b4f78d515e53cf74d58b2-Abstract-Conference.html)；NeurIPS 2025 Main Conference | Qwen2.5-1.5B/7B；WebShop | 无std归一化：1.5B GRPO SR56.8→67.4；7B66.1→75.2。 | 表1，第8页。同批相同状态分组，无额外critic；3B是QA实验。 |
| [SHADOW](https://ojs.aaai.org/index.php/AAAI/article/view/39570)；AAAI 2026, 40(28), 23935–23944 | Qwen2.5-1.5B/7B；WebShop | 动态感知分组：该文1.5B GiGPO SR71.4→76.6；7B76.3→81.5。 | 表1，第5页。这些是该文基线，不与GiGPO原表混算。 |
| [ActRe](https://ojs.aaai.org/index.php/AAAI/article/view/40794)；AAAI 2026, 40(41), 34906–34914 | Qwen2.5-1.5B/7B；WebShop | 1.5B GRPO SR57→64；只升全局温度却为56。 | 表2，第6页。完整方法是两阶段动作/思考训练，非只调温度。 |
| [DEPO](https://ojs.aaai.org/index.php/AAAI/article/view/40279)；AAAI 2026, 40(36), 30279–30287 | Qwen2.5-7B；WebShop | BC+KTO→DEPO：SR54→56；报告token886→726。 | 表1，第5页。3B只是未训练基线；含教师MCTS、BC和偏好训练。 |
| [Step-level Q Models](https://ojs.aaai.org/index.php/AAAI/article/view/34924)；AAAI 2025, 39(25), 27161–27169 | Phi3-mini3.8B等＋Phi1.5 Q模型 | Phi3的平均reward0.30→0.61；每一步对多个动作评分。 | 表2，第6页。不是61% SR；100题、10步，增加推理模型。 |
| [DAPO](https://papers.nips.cc/paper_files/paper/2025/file/a4277440d50f1f15d2cb4c14f7e0c0d2-Paper-Conference.pdf)；NeurIPS 2025 | Qwen2.5-32B Base；数学 | 动态采样、非对称clip、token损失和超长处理组成训练配方。 | 表1为逐项累积实验。无WebShop；各项增益不可直接独立相加。 |
| [LUFFY](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a9d5c33e38442450f9781bd7bad3c81e-Abstract-Conference.html)；NeurIPS 2025 | Qwen系列；数学 | 混合外部推理轨迹与自身rollout，并处理行为策略差异。 | 附录E.2表3有1.5B结果。无WebShop；不是直接复制教师轨迹。 |
| [UFT](https://proceedings.neurips.cc/paper_files/paper/2025/file/92e886487a8354b03d8bf4416eae6d7d-Paper-Conference.pdf)；NeurIPS 2025 | Qwen2.5含3B、Llama；数学/逻辑 | 平滑缩短提示前缀，并联合监督与强化学习。 | 第3.1.1节。不是仅减小指导比例；无WebShop实验。 |
| [Beyond the 80/20 Rule](https://papers.nips.cc/paper_files/paper/2025/hash/a797c2d2e0c1fdabf4d1ab8cd0b465c6-Abstract-Conference.html)；NeurIPS 2025 | Qwen3-8B/14B/32B；数学 | 只更新高熵20% token；8B近似保持效果，更大模型收益较强。 | 第2–4节。梯度mask与当前文档引导采样是不同方法。 |
| [SEED〔预印本〕](https://arxiv.org/html/2607.14777v1)；arXiv 2026; not verified as a paper in the four requested venues | Qwen2.5-3B-Instruct；WebShop128题 | 同文GRPO SR63.3→SEED78.9；训练后撤掉Skill。 | 表1。先训练事后技能分析，再联合RL/监督；非四会议已核收录。 |
| [UOPD〔预印本〕](https://arxiv.org/html/2609.34036v1)；arXiv 2026; not verified as a paper in the four requested venues | Qwen2.5-3B＋任务训练7B教师；WebShop128题 | OPD SR66.9→69.3；更重干预50%→30%时只有59.4。 | 表1/2。置信度选干预并监督纠正动作；非四会议已核收录。 |

不能直接用于结论的旁证还包括：[WebAgent-R1](https://aclanthology.org/2025.emnlp-main.401/) 属于 EMNLP 2025，3B结果来自 WebArena-Lite；[Dr. GRPO](https://openreview.net/attachment?id=5PAF7PAY2Y&name=pdf) 属于 COLM 2025，而且当前配置已关闭优势标准差归一化。它们不能冒充本轮四个会议中的新增WebShop证据。

文献之间也有可用于论文选题的共同问题：帮助学生采样成功轨迹，和让学生在撤掉帮助后保留能力，是两个不同目标。后续最有价值的机制证据是“文档事实正确 → 对应状态采取更好动作 → 有效梯度更新 → 裸开发集提升”的逐段检验。应将文档交付率、正确动作覆盖、非零优势、实际训练贡献和最终成功率放在同一条证据链上。
