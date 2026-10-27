# WebShop 分臂比例实验的云端运行说明

本实验只检验引导采样比例：control 为 8 条自主加 8 条引导，candidate 为 12 条自主加 4 条引导。两组均从原始 Qwen2.5-3B-Instruct 开始。固定 seed=42、T=133、epsilon=0.05、二值成功奖励和每题 16 条轨迹。两组短训练各固定 60 次更新，完成后停止。

模型、数据、服务、训练依赖和正式实验均部署到云端。电脑仅用于代码和文档编辑。云端根目录为 `/root/trace-webshop-arm-ratio`，仓库位于其 `code/` 子目录。2026-10-18 检查的矩池云实例有 4 张 A100 40GB。用户最新取消了时间截止，旧的明早八点及用卡时间限制不再作为本轮自动停止条件。

## 当前状态：2026-10-19 09:21:43，control pilot 因上游 403 停止

最新用户指令取消所有时间截止，“什么时候跑完什么时候停”。两臂仍各固定 60 步，完成评测和清理后结束；不增加训练步数，不使用虚假的遥远截止日期。用户此前已明确“不设置预算限制”，软件 API 金额封顶也已取消，费用记录继续保留。停止新增历史评测审计并继续原训练流程的授权保持有效。

最新云端快照为 `reports/progress-20261019-0924.json`，内部实际检查时间为 `2026-10-19T09:21:43.434271+08:00`；文件名不代表检查时间。`ws_control_pilot_20261018_p1` 已完成第 **1–14 步 actor 更新**，于 **2026-10-19 03:27:02（UTC+08:00）** 因研究员 API 上游 HTTP 403 终止，运行状态为 `INCOMPLETE`。队列为 `STOPPED`，candidate pilot 尚未启动。owner 已核验 `gpu_release_verified=true`，四张 GPU 均为 0 MiB、0% 利用率。代理 health 为 `stop_requested`，金额和时间限制仍为 `budget_enforced=false`、`deadline_enforced=false`；本次停止源于上游拒绝，不能归因为已取消的本地金额或时间限制。

**09:18:08 的单次直连诊断**同样返回 HTTP 403，错误码为 `AccessDenied.Unpurchased`，表示当前模型的使用资格被拒绝。该证据不能直接证明账号欠费。诊断报告为云端 `reports/upstream-403-diagnostic-20261019.json`。本次只更新状态，不修改训练代码或配置，不重启作业。

正式 control pilot 的初始 dev0 已完整核验，报告为 `reports/control-pilot-dev0-20261019.json`：256 题 × 3 次，保留 768 条，`complete=true`；成功 **43 / 768**，成功率 **5.5989583%**（原值 `0.055989583333333336`），平均 score 为 **0.21951187845719095**。ENV_ERROR 为 0，身份隔离记录为 0，审计问题列表为空。该结果仅对应训练前基线；没有 step20、40、60 评测，不能从已完成 14 步推断训练效果或两臂优劣。

**没有可恢复的第 14 步权重。** 已检查实际 run 下的 `checkpoints/`、云端根目录的 `checkpoints/` 和 `code/checkpoints/`，均不存在；`bounded_launch.log` 也没有 `Saving checkpoint to` 记录。正式 pilot 每 20 步保存一次，停止发生在首次计划保存之前。actor 更新记录和 rollout 日志保留了运行证据，但不能代替模型权重。

## 第 14 步训练批次结果（2026-10-19 核验）

云端报告为 `/root/trace-webshop-arm-ratio/reports/control-step14-result-20261019.json`。该批次属于 control pilot（每题 8 自主＋8 引导），包含相同的 32 个训练任务、512 条实际入训轨迹。已按 `sample_id + attempt_id` 将入训账本与 rollout 一对一匹配，验证二元 reward 与 won、score 一致；审计状态为 `ok`，完整组为 32，actor 更新返回标记匹配。

| 分组 | 成功数 / 样本数 | 完全成功率 | 平均 score × 100 | 非零优势样本数 | 零优势比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 自主 | 74 / 256 | 28.90625% | 55.22643849206349 | 200 | 21.875% |
| 引导 | 74 / 256 | 28.90625% | 61.39094629329004 | 74 | 71.09375% |
| 合计 | 148 / 512 | 28.90625% | 58.30869239267676 | 274 | 46.484375% |

两组记录的环境错误、截断和格式错误动作均为 0。引导权重 `teacher_alpha=0.9729086208503173`；自主、引导的优势绝对值之和分别为 `91.42857211828232`、`41.22700324654579`。非零优势样本包含正、负优势，不等于成功样本数，也不直接等于参数更新贡献。

全部 512 条的策略版本范围均为 `(13, 13)`：这些是第 14 次更新所用的训练轨迹，由完成 13 次更新后的策略生成，不是完成第 14 次更新后的独立开发集成绩。该批次引导的连续分数高约 6.16 分，但完全成功率相同；单个训练批次不能证明泛化提升，也不能与初始 dev0 的 5.60% 直接相减。原始入训证据位于运行目录 `training_audit/step_00000014_941c1f6c3e2d49f7a6ea4ca7cb6723c5/`，轨迹位于 `rollouts/step_00014/`。

### 第 14 步成功率相同的逐题核查

云端明细报告：`/root/trace-webshop-arm-ratio/reports/control-step14-task-comparison-20261019.json`。32 个任务各有两组各 8 次采样。引导成功次数更多的任务为 10 个，合计多 19 次；更少的任务为 13 个，合计少 19 次；相同为 9 个。因此两个 74 次是任务间增减抵消，不是逐题结果相同。代码没有强制两臂成功率相等。

引导臂 256 条中，有 64 条（8 个任务，占 25%）缺少文档，全部有 `research_error=1`，实际 `steer_count=0`；其余 192 条均有正的倾斜计数，并记录过 draft/final token 改变。自主臂 256 条均没有倾斜或 token 替换。缺文档的 64 条引导身份样本实际退回自主生成；研究失败的具体原因尚未逐条追溯，不能直接等同于后来导致整轮停止的 403。此前的“环境错误为零”不包含这类研究员文档生成错误。

两臂各 182 条未完全成功样本，其平均连续分数分别为自主 `0.3702180359323216`、引导 `0.4569275962133105`。这解释了完全成功数相同而总连续分数不同。当前二元奖励将这些未完全成功样本统一记为 0。引导的 182 条零优势正好对应这 182 条失败，不应将 71.09% 零优势解读为 71.09% 没有触发引导。以上是单批描述性证据，不足以确定缺文档是成功率相等的唯一原因，也不能证明泛化效果。

## 历史状态：2026-10-18 23:39:35，两次 smoke 完成，control pilot 当时正在运行

当时的精确快照保存在云端 `reports/progress-20261018-2339.json`，检查时间为 **2026-10-18 23:39:35（UTC+08:00）**。当时队列为 `RUNNING`，运行 `ws_control_pilot_20261018_p1`，已完成的阶段为 control smoke f 和 candidate smoke g。f 于当日 **22:29:22** 正常结束（原始 UTC 时间 `14:29:22.517961`），g 于当日 **23:33:25** 正常结束（原始 UTC 时间 `15:33:25.607527`）。g 的结束时间以此精确记录为准；此前读数 23:23 已更正。两次 owner 均为 `process_exited`，第 1、2 步 actor 更新均有返回证据，`gpu_release_verified=true`。

两次 smoke 的完整开发评测如下。每行均为 256 题 × 3 次，共 768 条；完整性检查通过，ENV_ERROR 均为 0。score 列是平均 score × 100，与成功率是不同指标。

| 运行 | 更新步数 | 成功数 / 轨迹数 | 成功率（%） | 平均 score × 100 |
| --- | ---: | ---: | ---: | ---: |
| control smoke f | 0 | 47 / 768 | 6.1198 | 23.0051 |
| control smoke f | 1 | 41 / 768 | 5.3385 | 22.2202 |
| control smoke f | 2 | 44 / 768 | 5.7292 | 22.3425 |
| candidate smoke g | 0 | 53 / 768 | 6.9010 | 23.2556 |
| candidate smoke g | 1 | 52 / 768 | 6.7708 | 23.5435 |
| candidate smoke g | 2 | 42 / 768 | 5.4688 | 21.0746 |

在 **2026-10-18 23:39:35** 这一历史时间点，control pilot p1 的 owner 仍在运行，`ended_at` 为空，actor 更新数为 **0**；初始 dev0 已生成 **223 / 768** 条记录，尚不完整，因此当时没有报告该批次的完整成功率或 score。candidate pilot 尚未启动。当日 23:38 的 GPU 快照显示四张卡利用率依次为 92%、78%、99%、100%，显存占用约 24 GB。两步 smoke 证明了更新、评测与清理流程能够运行；这些小规模结果不能判定引导比例的最终效果，仍需两臂完整 60 步及预注册比较。

以下保留此前部署与停止事件，均按各自时间点解释。

北京时间约 21:18，云端曾启动 proxy v2（PID 91581），关闭金额封顶。费用迁移已在云端完成，保留全部 320 条历史记录；按已核对的费率重估，保守占用从 169.55008 元变为 **29.720426 元**，证据为 `reports/api-rate-migration-v2.json`。这是迁移时的代理保守估算，不是供应商账单或当前累计费用。旧停止标记已保留。

为移除已加载进程中的旧截止条件，smoke d 已受控中断。2026-10-18 21:23:14（UTC+08:00），owner 记录 `reason=interrupted`、`gpu_release_verified=true`，GPU 显存为 0 MiB。该中断运行及此前记录全部保留，不能记为完整 smoke。

无截止的 proxy v3 已在云端部署（PID 111465），启动证据为 `reports/api-proxy-v3-start.json`；health 明确 `budget_enforced=false`、`deadline_enforced=false`。control smoke f 于 **21:27:52（UTC+08:00）** 启动，owner PID 为 **111472**，状态目录为 `/tmp/twcs-20261018-f`；已核验 `deadline=null`、`deadline_enforced=false`。启动原始时间记录为 `2026-10-18T13:27:52Z`。

自动衔接脚本已在云端通过 SHA 核验（前缀 `86e9`）。sequence v2 于 **21:28:30（UTC+08:00）** 启动，CPU 进程 PID 为 **116548**；当时状态为 `WAITING_CONTROL_SMOKE`，证据为 `reports/sequence-20261018-v2/status.json`。其时间策略同样为 `deadline=null`、`deadline_enforced=false`。序列先等待 f，通过后依次衔接 g 与 p1 两臂，各阶段均使用 `--no-deadline`。该时间点尚没有 actor 更新成功的结论。

**21:29:36 的实际复核**：f 的 owner PID 111472 仍未结束，队列仍为 `WAITING_CONTROL_SMOKE`；API health 为 `ready`，金额和时间限制均为 false。四张 GPU 显存占用依次为 5337、5125、5163、10065 MiB，利用率为 99%、99%、98%、42%；正在进行 SGLang CUDA graph 初始化，已发生实际 GPU 计算。f 的 dev0 记录数和 actor 更新数均为 0，因此此时只能记录加载初始化，不能记为评测或训练更新完成。

19:01:08（UTC+08:00），control smoke c 因 `research-abort.json` 预算保护自动停止。已核验 actor 更新数为 0；外层状态为 `INCOMPLETE`、`reason=abort_file`，停止原因是 `budget_exhausted`、HTTP 403。owner 已退出，`gpu_release_verified=true`，`remaining_owned_gpu_processes=[]`，当时 API health 为 `stop_requested`。两步 smoke 未完成。后续运行使用新的 run 和停止标记，保留旧运行、旧停止证据和累计费用账本；不能把 smoke c 改记为完成。用户断开本地网络不影响已经启动的云端后台作业。

21:04 的复核确认四张 GPU 均为 0 MiB、0% 利用率，smoke c 的 actor 更新仍为 0。该运行的 `step_00000_val` 有完整 768 条记录；`step_00001` 仅有 56 个 rollout 文件，未完成。这是 smoke d 启动前的历史状态。

smoke c 的初始开发集评测 dev0 已完整完成：256 个任务、每题 3 次，共 768 条记录；无重复、缺失或 ENV_ERROR。成功率为 **6.640625%**，平均 score 为 **0.2278842930796056**。该历史 dev0 的 768 条记录已通过云端完整性 gate 交叉验证。这是训练前基线，不能作为训练提升或两臂比较结果，也不能代替 f 的完成证据。

历史汇总 `probe/val_scan_out.csv:108` 记录 PGS 原始模型 step0 成功率 **4.00%**（60/1500，500 题），`:118` 的 **80.73%**（1211/1500）来自训练 **step200**。smoke c 的 6.64% 是 dev0，不能与历史 step200 直接比较训练效果。历史 test500 与当前 dev256 也不同；用户已取消继续审计，环境和完整条件的等价性尚未核验完毕。

## 项目状态：2026-10-19 09:21:43

| 项目 | 已确认 | 尚待完成 |
| --- | --- | --- |
| 训练环境 | 原依赖与 GPU 检查通过，两次 smoke 完成；control pilot 停止后 owner 释放验证通过，四张 GPU 均为 0 MiB、0% | 完整 60 步运行及最终比较；本次不重启 |
| WebShop 环境 | 1000 商品、6910 任务服务与依赖检查通过；两次 smoke 的六批完整评测，以及正式 control pilot dev0 均无 ENV_ERROR | 后续正式评测尚未发生 |
| 数据 | 已生成 6154 个训练任务、256 个开发任务和 500 个测试任务，身份互斥与哈希检查通过；正式 pilot dev0 保留 768 条，完整且无身份隔离或审计问题 | 保留冻结 manifest 和原始评测记录 |
| 学生模型 | 原始 Qwen2.5-3B-Instruct 文件已核验；control pilot 有第 1–14 步 actor 更新返回证据 | 没有 step20、40、60 评测，不能判断这些更新后的最终表现 |
| checkpoint | 实际 run、云端根目录及 code 下的 checkpoints 目录均不存在，日志没有保存记录；每 20 步保存一次 | 没有可恢复的第 14 步权重；更新账本不能代替权重 |
| 研究员 API | 金额封顶与时间截止均关闭；03:27:02 上游 403 导致停止；09:18:08 直连返回 `AccessDenied.Unpurchased` | 当前模型使用资格被拒绝，原因尚未进一步确认；不能断言欠费 |
| 费用记录 | 320 条原账本记录已按已核对费率迁移，迁移时保守占用估算为 **29.720426 元**；旧停止标记与迁移报告保留 | 继续保留累计记录并核对供应商账单；不能把保守估算当作实际账单 |
| 实验 | f、g 两次 smoke 完成；control pilot 更新至第 14 步后为 `INCOMPLETE`，队列 `STOPPED`；正式 dev0 成功率 5.5989583%，score 0.21951187845719095 | candidate pilot 未启动；两臂均无完整 60 步结果，不能进行预注册的 step40/60 比较 |
| 迁移与保存 | 静态 v2 资产归档于 20:19 完成，status 为 complete、exit code 为 0；文件导出与恢复流程成功 | 独立云端保存副本尚未建立；完整运行环境重建及训练复现未完成 |

FlashAttention 与 spaCy 的官方发布资产均已核对 SHA 并安装。训练 bootstrap 和 SGLang 原始模型探针完成后均已确认 GPU 清理。学生模型 `MODEL_MANIFEST` 记录的 snapshot 为 `a226d5576d7b3b7386a127f63bb30840e2d14acdb1f62360dc842132a4c0241a`。

首轮 control smoke 因 Ray 的 AF_UNIX socket 路径超过 107 字节，在训练前失败。第二轮使用 `/tmp/twcs-20261018-b`，18:39 因 CUDA 预加载冲突在训练前以 `returncode=1` 退出。原路径下已复现 `cudaGetDriverEntryPointByVersion` 错误；把训练虚拟环境的 `nvidia/cuda_runtime/lib` 放在 `LD_LIBRARY_PATH` 最前后，子进程探针通过，证据为 `reports/cuda-preload-probe.json`。前两轮均记录 `gpu_release_verified=true`。

第三轮于 18:47 启动，状态目录为 `/tmp/twcs-20261018-c`，run 名称为 `ws_control_smoke_20261018_c`，随后触发上述预算保护。已核验第三轮 GPU 释放和零 actor 更新。这些失败或未完成记录均不能计为已完成的 smoke。

WebShop 官方 Google Drive 及其内容下载域名发生连接超时。现已取得并核验官方 human 指令与固定社区镜像中的商品数据，共三个数据文件。镜像来自 `sparklabutah/timewarp-env-data` 的固定提交 `e6b1db309d01ca37cb52a0f8bc4537e798456b63`。来源与文件哈希应随数据保存；这些检查不证明它与历史实验的数据、环境版本完全相同。

旧预算代理最多预约 **190 元**，曾在 200 元总上限内留出 10 元余量；余额不足下一次预约时即拒绝。169.55008 元和其中 45.34578 元均为该旧规则下的上界，不能称为供应商实际费用。用户随后取消金额及时间上限：新代理显式使用 `--no-budget-limit --no-deadline`，保留原 SQLite 账本、费用审计和协议保护。19:01 的预算停止记录保持不变。

21:04 费用审计记录：原账本仍为 81 笔 reserved（124,204,300 微元）和 239 笔 settled_upper（45,345,780 微元）；已返回 usage 的输入 token 为 2,094,795，输出 token 为 57,498。按已核对的输入 2 微元/token、输出 12 微元/token，并将未知预约按原上界的五分之一保留，**只读模拟**重估为 29.720426 元。这不是供应商账单；原账本尚未迁移，预算门与 abort 标记未重置，也未据此恢复训练。

代理的 `--stop-file` 与作业包装器的 `--abort-file` 必须指向同一绝对路径，以下用 `TRACE_API_STOP_FILE` 表示该实际路径。取消金额及时间限制后，仍保留协议错误和上游 403 的停止保护；包装器发现停止标记后清理所属进程，并记录 `reason=abort_file`、`result_status=INCOMPLETE`。新运行应使用独立停止标记，旧标记归档保留；原 SQLite 费用账本不能通过重启清零。异常停止运行的 checkpoint 不能计作完整实验臂结果。

当前资产仍只保存在该实例的根盘。根盘上的目录或 tar 文件不是独立备份。只有资产包已保存到另一处可独立保留的云端存储，并在那里核对完整文件和 SHA256，才可记录“已独立保存”。本手册不记录密钥或带认证信息的 URL。

静态 v2 归档实际工作目录为 `/root/trace-static-assets-v2-20261018T110612Z-uyf7ciwf`。完成时间记录为 `2026-10-18T12:19:15.752858+00:00`，即北京时间 20:19；status 为 complete、exit code 为 0，文件导出与恢复流程成功。该结果仅确认此次静态资产操作，不能据此声称已独立备份或已重建可运行的训练环境。

## 部署前提

云端需要本仓库（含 vendored verl）、与旧运行兼容的训练环境、Qwen2.5-3B-Instruct 权重、原始 WebShop train/test Parquet、WebShop 服务，以及独立的研究员 API。配置中的研究员模型名是 `qwen3.6-35b-a3b`，必须核实服务实际模型版本，不能只改 API 别名冒充相同模型。换成 AWQ 量化模型需要另立实验。

[阿里云百炼官方模型页](https://www.alibabacloud.com/help/en/model-studio/qwen3-6-35b-a3b)列出了该模型及 Function Calling 支持。当前接入已经通过非思考模式工具调用探测；两臂启动时仍须使用同一云端服务和预算代理。官方 API 模型名称一致，不代表与历史自托管服务的权重版本和推理实现已经核验一致。

使用独立云端目录和虚拟环境。保留节点现有环境。以下命令只在云端执行；激活脚本从云端私密位置读取密钥，不在代码或报告中写入密钥：

```bash
export TRACE_ROOT=/root/trace-webshop-arm-ratio
source "$TRACE_ROOT/code/scripts/cloud/activate_experiment.sh"
cd "$TRACE_ROOT/code"
```

激活脚本从根目录派生模型、数据和缓存路径。WebShop 地址为本机回环服务，研究员地址为本机预算代理；启动训练前仍须确认两项服务已就绪。准备和分析脚本只需要 Python 标准库与 PyArrow；训练必须使用项目所需的 GPU 环境，不能把本地诊断测试环境当作训练环境。

上传代码时必须保留 `train/agent_loop_config.yaml` 和 `verl/verl/trainer/config/data/legacy_data.yaml`。不能按路径中出现 `data` 就排除：后者属于 Hydra 的必需配置。训练入口在仓库下使用 `runs/`。当前 `code/runs` 是实际目录，根目录 `runs/` 为空，尚未建立软链接；按当前布局启动会写入 `code/runs`。若以后统一到根目录 `runs/`，必须先保留并迁移已有内容，再建立和核验链接，不能直接覆盖。

## 固定开发集

在云端项目根目录执行：

```bash
python scripts/classic/prepare_webshop_arm_ratio.py \
  --source-train /cloud/path/original/train.parquet \
  --test-file /cloud/path/original/test.parquet \
  --output-dir "$TRACE_WEBSHOP_ARM_DATA"
```

脚本按 `sha256("42\0" + task_id)` 排序，划出 256 个开发任务。两组共享相同的 train/dev 文件。test 只做身份交集检查，不读取评测结果来选择方案。重复任务、重复 goal_idx、schema 错误和集合重叠会停止准备。输出目录必须是新目录。manifest 记录文件及任务身份哈希。

## 检查和固定步数运行

用户已授权继续以下原定流程，取消时间截止和金额封顶。执行前核验代理两项限制均关闭、旧停止证据已保留、当前作业使用新的停止标记和独立运行目录。新 smoke 通过之前，不把旧的失败运行当作 pilot 准入证据。以下命令仅说明调用格式，已有作业或自动序列运行时不要重复手工启动。

先检查配置。`launch_train.py --dry-run` 不联网、不建 run 目录、不启动训练：

```bash
python scripts/launch_train.py \
  --config configs/experiments/webshop_arm_ratio/control.yaml --dry-run
python scripts/launch_train.py \
  --config configs/experiments/webshop_arm_ratio/candidate.yaml --dry-run
```

每臂先运行两步冒烟。启动前先对内层入口使用 `--dry-run` 检查云端输入文件、哈希和端点格式。实际执行时，**必须用 `run_owned_job.py` 包住训练入口**，让退出后的 Ray 子进程也进入本作业的清理流程。两层都显式传入 `--no-deadline`；run 名称和外层状态目录必须全新。`--no-deadline` 与 `--deadline` 互斥，不能省略时间策略，也不能同时传入。

外层状态目录使用短且唯一的 `/tmp` 路径。Ray 会在包装器给出的路径后追加目录和 socket 文件名；不要把完整项目路径或长 UUID 作为状态目录。b、c、d、f 的状态目录均已使用，不能重用。下面仅用 h 示范格式，不是额外运行计划；任何执行前均须确认名称和状态目录未被使用或序列预留：

```bash
export TRACE_JOB_STATE=/tmp/twcs-20261018-h
export TRACE_JOB_RUN=ws_control_smoke_20261018_h
python scripts/cloud/run_owned_job.py \
  --state-dir "$TRACE_JOB_STATE" \
  --gpus 0,1,2,3 --no-deadline \
  --abort-file "${TRACE_API_STOP_FILE:?设置为预算代理 stop-file 的同一绝对路径}" \
  --cwd "$TRACE_ROOT/code" -- \
  python scripts/classic/run_webshop_arm_ratio_bounded.py \
    --phase smoke --arm control --run-name "$TRACE_JOB_RUN" \
    --no-deadline
```

candidate 使用对应的独立 run 名称、外层状态目录和 `--arm candidate`。两臂按次序使用这些 GPU；不能同时运行两条占用同一组 GPU 的命令。单次冒烟入口不自动启动短训练；另由序列脚本在核验阶段证据和 GPU 释放后衔接下一阶段。完整开发评测与 60 步训练完成后自然结束，失败时停止并保存证据。

Pilot 必须提交同臂已成功的 smoke 目录。入口核验同模型、seed、配置与数据哈希，以及第 2 步 actor 更新返回证据。完整 dev 评测仍须检查；包装器不声称已核验其完整性。

```bash
python scripts/classic/run_webshop_arm_ratio_bounded.py \
  --phase pilot --arm control --run-name ws_control_pilot_20261018 \
  --smoke-run runs/CONTROL_SMOKE_THAT_PASSED \
  --no-deadline --dry-run
```

上段仅做 pilot 配置检查。实际 pilot 同样从外层包装器进入，不能直接删除上段的 `--dry-run` 后裸跑：

```bash
export TRACE_JOB_STATE=/tmp/twcp-20261018-a
export TRACE_JOB_RUN=ws_control_pilot_20261018_a
python scripts/cloud/run_owned_job.py \
  --state-dir "$TRACE_JOB_STATE" \
  --gpus 0,1,2,3 --no-deadline \
  --abort-file "${TRACE_API_STOP_FILE:?设置为预算代理 stop-file 的同一绝对路径}" \
  --cwd "$TRACE_ROOT/code" -- \
  python scripts/classic/run_webshop_arm_ratio_bounded.py \
    --phase pilot --arm control --run-name "$TRACE_JOB_RUN" \
    --smoke-run runs/CONTROL_SMOKE_THAT_PASSED \
    --no-deadline
```

单 epoch 的 smoke 至少需要 64 个训练任务；pilot 至少需要 1920 个。数据不足时停止，不能把少于 60 步的运行标成完成。

内层入口只终止自己创建的进程组，无法单独证明 Ray worker 已释放。外层 `run_owned_job.py` 设置本地 Ray 地址，用 Linux 子进程收养和进程身份记录跟踪本作业后代，并通过 pidfd 清理已证实属于本作业的进程。无截止模式仍会在普通退出、异常或外部中断时执行清理；不会执行 `ray stop`、全局 `pkill` 或依据 GPU 占用杀掉其他任务。旧的显式截止模式仍兼容，但本轮不使用。

结束后检查外层状态目录中的 `status.json`、`processes.json` 和 `command.log`，并保存非秘密证据到 `reports/`。无截止模式写入 `deadline=null`、`deadline_enforced=false`，外层 `finished_before_deadline=null`，不能把该空值解释为失败或声称按时完成。只有进程正常成功退出、外层 `gpu_release_verified=true` 且没有未清理的本作业进程，才能记录完整成功；异常退出后的 GPU 清理成功也不代表训练完成。内层进程退出不是释放证据。watchdog 被强制杀死、主机故障或驱动挂起时仍可能无法完成清理，状态必须保留未验证，不能据此启动下一作业。

每次作业无论成功或失败，都应及时保存 `/tmp` 中的状态证据。先确认 `TRACE_JOB_STATE` 与 `TRACE_JOB_RUN` 对应同一次启动，再把外层 `status.json` 复制为该 run 下的 `owned_job_status.json`，并保存进程证明及状态路径关联。以下归档命令只在包装器已经退出、实际 run 目录已存在时执行；不覆盖既有证据：

```bash
(
  set -eu
  run_dir="$TRACE_ROOT/code/runs/$TRACE_JOB_RUN"
  test -d "$run_dir"
  test ! -e "$run_dir/owned_job_status.json"
  mkdir "$run_dir/owned_proofs"
  cp "$TRACE_JOB_STATE/status.json" "$run_dir/owned_job_status.json"
  cp "$TRACE_JOB_STATE/processes.json" "$run_dir/owned_proofs/processes.json"
  printf '%s\n' "$TRACE_JOB_STATE" > "$run_dir/owned_proofs/state_directory.txt"
)
```

`command.log` 检查后另存为非秘密运行证据。如果作业在建立 run 目录前失败，将状态和日志保存到独立的 `reports/` 失败记录，并标明本次 job 的短状态路径和预定 run 名称。分析器会检查 `owned_job_status.json` 与内层 `bounded_stop_status.json`；异常结束、INCOMPLETE 或损坏的状态证据均阻止 PASS。首轮 smoke 还须检查 `research_error`、`research_doc_missing`，避免普通研究请求失败被当作正常引导。

## 数据质量与训练证据

- 每轮最多 1024 个生成 token，最多 15 次动作和 15 次回复，总流预算 16384。后端返回超限内容时，在执行动作前记为基础设施错误；不静默裁剪后继续执行。
- 每条轨迹记录 task、uid、session、sample、attempt 和 evaluation 身份。独立采样得到相同内容只标记。重复保存的同身份副本去重留审计；冲突身份隔离。评测重复或缺样本时不能通过实验判定。
- 时长和长度离群值用 median + 6 MAD 标记；MAD 为零时使用 IQR 回退。有效的稀有成功、失败及 0/1 奖励保留。统计离群标记不改变主指标。
- `episode_attempts` 保留学生尝试，包括失败和重试。`research` 档案按 session 计一次，避免一份文档在多个 rollout 中重复计费。缺失的 API usage 标为未知。返回 token 和并发墙钟不等于完整 GPU 计算成本；教师评分成本与 GPU 时间仍需云端计量。
- `training_audit/step_<N>_<audit_id>` 保存实际选入批次的分组、奖励、优势、有效 token 数及分臂汇总。只有 actor 更新调用成功返回后才写 `actor_update.json`。这不表示每行都产生非零梯度。

上述改动不改变采样核、优势公式或有效样本的奖励。

## 判定

第 0、20、40、60 步在同一开发集评测，每题 3 次、温度 0.6，学生不使用文档和教师。完成后在云端运行：

```bash
python scripts/classic/analyze_webshop_arm_ratio.py \
  --control-run runs/CONTROL_PILOT \
  --candidate-run runs/CANDIDATE_PILOT \
  --manifest "$TRACE_WEBSHOP_ARM_DATA/manifest.json" \
  --steps 40 60 --bootstrap-seed 42 --bootstrap-samples 10000 \
  --output-dir reports/webshop_arm_ratio
```

报告包括 `report.json`、`report.md` 和 `audit.jsonl`。第 40 步成功率点差必须为正；第 60 步点差及按任务配对 bootstrap 95% 区间下界必须为正。样本、配置或数据隔离证据缺失时返回 INCOMPLETE，不能扩大训练或声称获得最终 benchmark 优势。

本地旧运行只有汇总证据；尚未恢复指定历史步骤的完整原始轨迹。新的配置和测试通过不代表新实验已完成，也不证明引导比例是唯一失败原因。

云端早先三批通过 54、25、10 项单元测试，共 89 项。随后分析器补丁的两文件 SHA 与预期完全匹配，22 项云端测试于 1.379 秒内通过；其脚本 SHA 前缀为 `1d74f558`，最新 CUDA 运行修正 SHA 前缀为 `9e5ca659`。本地另有 7 项测试的云端同步尚未确认，待执行命令位于 `/private/tmp/trace-tests-sync-python.command.txt`，不能将它们记为云端通过。不同批次不相加作为一次完整测试。截至 23:39:35，两次两步 smoke 已完成，完整 60 步训练尚未完成。

## 云端资产清单与独立保存

完整清单应以云端根目录为扫描根，覆盖以下真实目录：

| 目录 | 必须保留的内容 |
| --- | --- |
| `code/` | 实际部署的源码、配置、测试和依赖声明，包括上述两项必需配置 |
| `models/` | 完整学生模型、分片索引、tokenizer，以及下载来源和校验证据 |
| `data/` | 原始 train/test 文件、冻结的 train/dev 划分和身份/文件哈希清单 |
| `environment/` | WebShop 固定版本源码、数据、Lucene 索引、源码与环境 manifest、重建所需说明 |
| `locks/` | Python 3.12 训练依赖锁定记录、Python 3.10 WebShop 依赖清单、运行时版本证据 |
| `code/runs/`；未来可迁至根目录 `runs/` | 当前实际运行目录所在位置，需保留日志、rollout、研究档案、实际入训账本和已有 checkpoint；checkpoint 文件以实际目录及清单核验为准，不能从更新次数推断完整保存；根目录 `runs/` 仍为空 |
| `reports/` | 配置与运行检查、费用账本一致性快照、评测报告和资产审核记录 |

资产工具自动排除 `.secrets/`、`.cache/`、`.venvs/`、目录内的 `venv/`、已安装环境和凭据文件。虚拟环境必须在目标机器重建，不能把复制的虚拟环境视为可迁移安装。历史配置和日志中的绝对路径仅记为 portability warning，原始字节与哈希保持不变；新运行路径由新的根目录派生。

`environment/` 下的 WebShop 源码、`data/` 和 `search_engine/indexes_1k/` 均在允许范围内；其中的 `venv/` 会被排除。若实际 WebShop prefix 在别处，须先把可复建资产归入允许目录，不能因清单执行成功就声称它已被收录。`runtime/` 不在当前工具允许范围内：需要保留的非秘密启动证据应另存于 `reports/` 或 `environment/`，激活脚本则按新根目录重建。软链接不跟随；清单应扫描其指向的允许范围内真实资产。

下载和安装完成后，先固定源码版本或部署标签，停止会修改待归档文件的作业。对运行中仍有变化的模型下载、索引或 SQLite 账本，先取得稳定文件或一致性快照。下面保留完整云端归档流程作为参考；本轮静态 v2 的实际完成状态单独记录于上文，不能把它扩大为下列独立保存流程已完成：

```bash
export ASSET_ROOT=/root/trace-webshop-arm-ratio
: "${TRACE_SOURCE_REVISION:?设置实际部署的版本号或审核标签}"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
audit_dir="$ASSET_ROOT/reports/asset-audit-$stamp"
mkdir -p "$audit_dir"
asset_tool="$ASSET_ROOT/code/scripts/cloud/asset_manifest.py"

python3 "$asset_tool" manifest --root "$ASSET_ROOT" \
  --category code --category configs --category models --category data \
  --category environment --category runs --category reports --category locks \
  --revision "$TRACE_SOURCE_REVISION" --output "$audit_dir/inventory.json"
```

`manifest` 不复制模型。先审查每类数量、文件字节数、排除条目和必需文件名单。`selection_complete=true` 只表示显式选择范围内没有未解决的排除；它不证明模型、数据或云端环境已经可运行。私密内容和无法读取的必需源码必须解决，不能靠缩小选择范围掩盖缺失。

目前尚未提供独立保存目标。下面的 `TRACE_RETAINED_DIR` 必须指向已确认能够独立保留的云端挂载位置，不能仅换成当前实例根盘的另一个目录；若使用对象存储，应另做上传和目标端校验。确认目标后才执行：

```bash
: "${TRACE_RETAINED_DIR:?设置已确认的独立云端保存目录}"
archive="$TRACE_RETAINED_DIR/trace-webshop-arm-ratio-$stamp.tar"
python3 "$asset_tool" export --root "$ASSET_ROOT" \
  --manifest "$audit_dir/inventory.json" \
  --category code --category configs --category models --category data \
  --category environment --category runs --category reports --category locks \
  --output "$archive"
python3 "$asset_tool" verify --archive "$archive"
sha256sum "$archive" > "$archive.sha256"
```

模型导出是显式选择，会读取和复制大权重；只在需要完整迁移包时执行。现有输出文件不会被覆盖。还需保存目标存储的留存证据，重建依赖并验证服务后，才能更新“可恢复运行”的状态。当前 **静态 v2 导出与恢复成功；独立保存、环境重建和训练复现尚未完成**。
