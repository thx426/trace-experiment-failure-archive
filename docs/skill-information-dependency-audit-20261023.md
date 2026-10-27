# Skill 信息依赖审计（2026-10-23）

结论：**现有流程有信息依赖检查缺口，真实文档也存在任务答案提示和实验得分信息；目前证据不足以把后期性能下降归因于它。** 已完成源码追踪、7168 条原始 rollout 与实际训练账本关联、逐时刻商品 ID 审计、案例人工复核，以及一个可选的保守采样防护。没有做新的 GPU 训练，也没有得到防护改善最终性能的结果。

本次开始于 10 月 22 日，因此证据目录保留原名 `reports/skill_information_20261022/`。

## 1. 当前究竟在训练什么

当前主线是 PGS/SPS 采样引导 + GRPO。学生初始前缀没有 Skill；教师前缀有 Skill，并影响学生 rollout 的采样分布。随后按学生上下文对生成的响应计算训练损失。因此，不是把研究员完整轨迹直接做 SFT，但用户指出的信息不对称仍然可能经采样进入学生训练。

源码证据：

| 环节 | 现有行为 | 对信息依赖的意义 |
|---|---|---|
| `research/prompts.py::DOC_RULES_TEMPLATE` | 已有 INFORMATION-STATE RULE，要求把发现改写成发现动作 | 是软提示；同时又允许在 WHEN/DO 中给出路径，只要再要求确认。知道去哪里读本身仍可能是额外信息 |
| `research/tools.py::_check_doc` | 检查长度、训练词汇、连续动作照抄、零成功却声称 verified | 没有学生可见上下文输入，不追踪信息来源；第三节也没有结构化强制校验 |
| `research/tools.py::_finalize` | 要求修改一次后，可接受仍带 flags 的文档 | flags 不是拒绝条件。复现测试中，隐藏路径文档完全不触发 flags；同一违规文档重复提交可被接受 |
| `train/agent_loop.py::_research` | 主要核验产物、归档、错误和版本 | 不判断文档事实是否可由学生获得 |
| `core/episode.py` | assistant 生成 token 的 mask 为 1，环境观察为 0 | 没有“该 token 是否依赖 Skill 私有事实”的掩码 |
| `verl/.../core_algos.py` | 引导臂按正向超额收益等规则计算优势 | 成功、正优势、分布距离限制和重要性权重均不等于信息可迁移性 |

WebShop 不注入 gold attributes（`gt_text=None`），但研究员见过的多次实验、实验得分、正确商品判断仍然超出该学生 episode 的观察。不能用“没有直接注入 GT”排除这种依赖。

本地与云端采集时的上述关键源码 hash 相同；`agent_loop.py` 的修改前快照也与云端相同。见证据 manifest。历史 gold-answer 字符串查重探针不能替代这个审计：它们既不完整覆盖信息类型，也没有逐行动检查可见时间。

## 2. 实际运行的审计实验

证据来自云端 `ws_control_pilot_20261018_p1` 的 step 1–14。每条 rollout 按 `(global_step, sample_id, attempt_id)` 关联实际训练行，按该 rollout 的 `research_archive` 取当时的文档，校验 task、step、文档 hash，并关联 actor update 的 audit ID。没有按 task_id 任取“最新文档”。

| 项目 | 结果 |
|---|---:|
| 原始 rollout / 实际训练行 | 7168 / 7168 |
| 唯一匹配数 | 7168 |
| 未匹配 / 读取失败 | 0 / 0 |
| 研究档案数 | 448 |
| 非空文档数 | 319 |
| 含商品 ID 的非空文档 | 294 / 319（92.2%） |
| 无可用文档的档案 | 129 |
| 每臂有非空关联文档的轨迹 | 2552 |
| 每臂有含 ID 文档的轨迹 | 2352 |
| 使用文档 ID 早于环境首次展示（动作或推理） | 两臂均为 0 |

这里的“0”只覆盖 ID 字面暴露，不是信息安全通过率。129 个无可用文档的档案没有被算作“通过”；它们对应每臂 1032 条缺文档轨迹。bare 臂关联文档只用于对照审计，不表示该臂实际看到了文档。

时间顺序规则：先核对本轮 action/thought，再加入本轮 observation；assistant 自己写出的字符串不算环境证据；疑似私有信息行动后的观察不用于洗白后缀。商品 ID 按大小写归一，路径保持大小写敏感。公开初始面只取真实 text scaffold 与初始观察，不把整个 adapter 源码或研究档案算作学生知识。

**统计修订记录：** 初版只匹配大写 ID，得到 194 份含 ID 文档，会漏掉小写动作；该版本已作废。归一大小写后重跑得到 294 份，提前使用计数仍为 0。再排除 adapter 源码中不属于 text scaffold 的工具例子后，初始不可见 ID 文档轨迹由每臂 2296 修正为 2352。本文和本地 `summary.json` 均为最终口径。

辅助搜索词筛查产生的 10 个样本主要是单复数、同义改写或重复既有错误，不能作为隐蔽搜索线索泄漏的阳性计数。

## 3. 比“有没有看到 ID”更重要的证据缺口

一个搜索结果列出某个 ID，只能支持“现在可以点击它”，不能自动支持“它满足尚未读取的材质属性”或“购买它一定成功”。以下是定向复核案例，不能据此估计全量语义问题发生率。

| 案例 | 文档或轨迹内容 | 实际训练账本 | 判断 |
|---|---|---|---|
| `webshop_03167`, step 6 | Skill 要求信任首个指定商品、不要因为标题 Pink 而继续寻找；学生在点击详情前断言该商品符合 stainless steel、classic grey | reward=1，advantage=0.746240139，230 个有效响应 token，steer_count=113，doc flags 为空 | 明确的语义证据缺口候选进入了正优势训练；并非提前复制未见 ID |
| `webshop_03637`, step 10 | Skill 指定镜子 ID；学生搜索后再点击，并在详情页看到尺寸选项后选择 | reward=1，advantage=0.493057966，307 个有效响应 token | 含具体 ID 不足以判整条轨迹不可迁移；选尺寸有公开依据 |
| `webshop_03708`, step 10 | Skill 明说评分接受 full，spot clean/easy clean 无须检查，因为三个选项即可得分 | 所选样本 reward=0，advantage=0，steer_count=430 | 文档确实携带评分实验知识；此失败样本不能充当正优势训练证据 |
| `webshop_04682`, step 12 | Skill 给出指定 HDMI 商品和完整操作顺序，并要求不检查中间状态；也包含有用的“显式选择尺寸和颜色”策略 | reward=1，advantage=0.490023941，226 个有效响应 token，steer_count=93，doc flags 为空 | 任务特定操作指令与可迁移策略混在一起，不能用成功标签一并认证 |

这些样本的正优势与非零有效 token 均来自实际 `advantages_computed` 账本，并有匹配的 actor update 标记，不是仅从 rollout 成功推断“已训练”。它们不证明具体参数更新量，也不证明某个错误推理由 Skill 导致；bare 模型本身也可能作出同样断言。

完整远端案例保存在 `/skill-info-semantic-cases-20261023.json`。本地 `case_excerpts.json` 保留 sample/attempt/audit ID、文档和 rollout hash、原文摘录与判断边界。

## 4. 已实现的防护及其边界

新增 `steering.information_guard=reject_unobserved_locators`。它在教师前缀构造之前，用真实学生初始 scaffold、初始 observation 和工具 schema 检查文档中的绝对 POSIX 路径、HTTP URL、WebShop ID。发现未公开的定位信息时，整份文档不参与该 episode 的引导，从开头按 bare 生成。即使文档写“读取该路径后确认”，仍会拦截。

这样避免了“先按秘密信息走到某个状态，事后只删除那几个 token，仍训练依赖它的后缀”的问题。已有旧轨迹需要从干净前缀重新 rollout，不能靠删除 Skill 文本修复。

实现文件：

- `src/trace_opd/analysis/information_dependencies.py`：字面信息筛查和按时间顺序审计。
- `src/trace_opd/train/agent_loop.py`：生成前的 gate、doc hash/候选项记录，以及 evaluated/blocked/candidates 计数；研究行补齐相同计数键，避免混合批次字段长度不齐。
- `src/trace_opd/common/config.py`：模式校验、开启时改变 objective fingerprint；不支持的 research-RL 路径明确拒绝开启，避免静默绕过。

实验臂仍保留原分配身份（intention-to-treat）；被拦截行的实际行为是 bare，由 blocked 计数与空 steering ledger 区分。它不代表研究员未花费实验成本。

**默认仍为 `off`，以保留历史对照；没有修改云端训练代码或启动中的训练任务。** off 继续记录候选项但不改变采样。历史 control fingerprint 保持 `fbd2b1d64d65`；开启防护后为 `c915d0e64337`。

这只是可复现的保守防护/消融，不是通用语义泄漏检测器：

- 单看 ID 就会拦下本次 294/319 份非空文档，包含不少可通过正常搜索发现的信息，会一起损失有用策略。真实完整 gate 的总拦截率应读取运行日志，不能只拿 ID 比例代替。
- 相对路径、符号名、答案数值、隐蔽搜索词、目标正确性、评分事实与“跳过验证”这样的语义依赖仍可能漏检。
- 合法输出路径和常见网址也可能误报；未发现字面问题不等于已证明可迁移。
- 本次只完成 CPU 测试和启动命令 dry-run，尚无 GPU 集成 smoke 或性能改善证据。

## 5. 应保留的学习内容与后续因果验证

建议将文档改为“可观察条件 → 发现动作 → 所需证据 → 决策”，而不是已知目标的操作指令。例如，将“点击 B09L44298X，忽略 Pink 标题并购买”改为：

> 用原始任务中的类别和属性搜索。选择标题和价格有希望的候选项打开详情；需要的属性尚未出现时读取 Features/Description。只在可见选项中选择任务要求的颜色和尺寸。将尚未核实的属性保留为未知；若确认不符，再检查下一个候选项。

应区分一般操作策略和实例事实。学会搜索、检查属性、显式选择选项、识别无效动作，是可迁移策略；提前知道哪个不透明 ID 正确、哪个未观察属性可以不查、评分器接受哪个冲突解释，是需要单独约束的实例知识。禁止一切新词或一切教师知识也不合理。

要判定“是否导致后期下降”，仍需同起点、同任务序列和预算下比较原文档、去除实例事实但保留策略的文档，以及 bare 对照；另外报告严格 gate 的丢文档比例，避免把“关掉大部分指导”误称为策略净化效果。关键指标是无 Skill 的 held-out/dev 曲线、发现/核验动作、无依据断言、信息依赖样本的优势权重和实际 token 保留情况。若发现受污染行动，修复实验应从该行动前的干净状态重新采样整个后缀。

本次能取得完整入训链路的 run 只到 step 14，并不是覆盖既往后期下降的完整原运行。`runs/classic_ws3b_pgs/` 本地只有配置、启动 manifest 和 fingerprint，不能据此还原晚期轨迹；control pilot 每 20 步保存一次，在 14 步后停止且无相应 checkpoint（见 `docs/webshop-arm-ratio-runbook.md`）。因此本次优先完成机制审计，没有把另开一轮长程训练当作原下降的根因证明，也没有声称问题已被完全解决。

## 6. 复现与产物

远端原始证据包：`/skill-info-evidence-20261022.json.gz`，167168955 bytes，SHA256 `2673622f4bcf15a61692a4fdadd8d60b6b535a8804c4a1514dc40f5970ef999d`。

去除巨型 steering 数组后的紧凑证据包：`/skill-info-compact-20261023.json.gz`，13249988 bytes，SHA256 `210ceaabbe1702780be0891ad292ab1a240cc8d3427eaa2cbe5c82b92563226f`。原包、紧凑包、审计脚本和完整结果均保留在云端；浏览器下载未完成。本地保存的是通过页面 DOM 读取的统计和案例摘录，不冒充完整原始数据副本。

远端实际执行脚本：`/skill-info-audit-20261023.py`。仓库提供相同匹配和逐时刻审计规则的 CLI，另增加了对重复/缺失关联的断言：

```bash
python3 scripts/classic/audit_skill_information.py \
  /path/to/skill-info-compact-20261023.json.gz \
  --out reports/skill_information_reproduced
```

保守 gate 的配置检查命令（需设置 control 的既有环境变量）：

```bash
PYTHONPATH=src python3 scripts/launch_train.py \
  --config configs/experiments/webshop_arm_ratio/control.yaml \
  --run-name webshop_info_guard_ablation \
  --set steering.information_guard=reject_unobserved_locators \
  --dry-run
```

验证已通过 **48 项测试**：19 项信息依赖/实际生成方法分支测试、4 项审计关联与时序回归测试、25 项既有采样配置/训练账本/episode 测试。实际生成方法测试用 AST 提取原方法并替换 GPU 后端，检查拒绝时不触达教师，以及 off/公开定位符仍触达教师构造；不能替代完整模型运行。

本地产物：`summary.json`、`case_excerpts.json`、`evidence_manifest.json`、`tests.txt`、`guarded_launch_dry_run.txt` 和修改前源码快照，均在 `reports/skill_information_20261022/`。
