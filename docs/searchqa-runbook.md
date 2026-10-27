# SearchQA × Qwen2.5-7B 双臂实验 RUNBOOK(CD-011,2026-08-13 定稿)

> 单一真源:从零启动/续跑/看护/观察的全部命令与参数依据。
> 配置真值在 `trace_opd-bench/configs/classic/searchqa_7b_{pgs,grpo}.yaml`;
> 本文只解释「为什么这么配」与「怎么跑」,数字以 yaml 为准。

---

## 0. 一句话

机制臂 `classic_sqa7b_pgs`(PGS:采样端保尾核倾斜 + 35B 外部 researcher 产
per-task 特权 doc)对照基线臂 `classic_sqa7b_grpo`(纯 GRPO,零特权信号),
A800 各 4 卡并行,batch128×n8 = 1024 episode/步,2 epochs ≈ 2650 步。

---

## 1. 前置依赖(缺一不可)

| 依赖 | 位置 | 验证命令 |
|---|---|---|
| **retriever**(faiss + e5,wiki-18) | 162:8000,pid 常驻、`--faiss_gpu` 占 4 卡各 ~10G | `curl -s -m10 -XPOST http://HOST_IP:8000/retrieve -H 'Content-Type: application/json' -d '{"queries":["test"],"topk":3,"return_scores":true}'` |
| **35B 双端点**(researcher + narrator) | 162:8880 / 162:8881 | `curl -s -o/dev/null -w '%{http_code}' -H "Authorization: REDACTED $TRACE_35B_KEY" http://HOST_IP:8880/v1/models` |
| 训练数据 | `/fs/fast/.../classic_searchqa_data/train_trainer.parquet`(169,615) | 见 §2 字段要求 |
| val 数据 | 同目录 `val_shuf2048.parquet`(test 集 seed42 抽 2048) | — |
| 编译入口 | `TRACE-OPD-PGS/env_pgs_sglang_a800.sh` | **发射脚本已内置 source,勿绕过** |

⚠ **retriever 进程绝不可杀**:它是一个进程占四张卡(索引 fp16 分片),SearchQA
的每次 `<search>` 都打它。

---

## 2. 数据契约(踩过坑,必须满足)

parquet 需含 `prompt / data_source / agent_name / reward_model / extra_info`:

- **`agent_name = "trace_opd"`** —— verl 据此选 agent loop(`train/agent_loop_config.yaml` 注册名)
- **`reward_model.style = "env"`** —— 声明 reward 由 AgentLoop 提供,**绕过 verl 的
  rule 分派**;若为 `"rule"`,verl 会去查 `reward_score` 注册表,而我们的
  `data_source`(`searchqa` / `searchqa/nq` …)不在其中 ⇒ `NotImplementedError`
- `extra_info` 需含 `task_id / task_type / split / question / golden_answers / initial_obs`

原始 `train.parquet` / `test.parquet` **不满足**(缺 agent_name、style=rule),
故用 `probe/sqa_fix_parquet.py` 产出 `train_trainer.parquet` 与 `val_shuf<N>.parquet`
(原文件保留)。判分不受影响:EM 内嵌 adapter、读 `extra_info.golden_answers`。

排查工具:`probe/sqa_audit.py`(逐项对照 ALFWorld 的活样本 schema)。

---

## 3. 关键配置与依据

### 3.1 交互协议:文本标签(**不用 FC**)

```
<think>推理</think> → <search> 查询 </search> → 环境回 <information>…</information>
                                              → <answer> 答案 </answer>
```

- **依据**:Search-R1(`scripts/data_process/nq_search.py`)与 SEED
  (`prompts/search.py`)代码级核实,两家都用文本标签且 prompt 内置 one-shot
  (`For example, <answer> Beijing </answer>`)
- **动因**:Qwen2.5(2024)对 function-call 遵循不足。FC 档实测 **11.8% 轨迹**
  因连续格式违规判 `DEGENERATE:format`(均每条 3 次),而 4 步预算下一次失误即 25%
- **实测效果**:文本档 2048 条 val **零违规 / 零截断 / 零 DEGENERATE**;EM 33.4%→35.3%
- 标签沿用原生 `<think>`(不同于 ALFWorld 的 `<reasoning>`):**已实测 Qwen2.5 的
  chat template 不剥离历史轮 `<think>` 块**(Qwen3 系才剥离)

### 3.2 规模口径

| 项 | 值 | 依据 |
|---|---|---|
| batch × n | 128 × 8 = **1024 eps/步** | 用户裁定;对齐 SEED 的 SEED 臂(128×8) |
| max_env_steps | 4 | Search-R1 / SEED 三家一致 |
| 步数 | `total_epochs: 2` ≈ **2650 步** | 用户裁定(不设步数上限) |
| val | test 集 seed42 固定 **2048 条** | 用户裁定("固定则取大");对齐 SEED 的 512 并放大 |
| val 口径 | **greedy**(temp 0) | 我方口径;SEED/GiGPO 系用 temp0.4 采样,**对拍须注明** |
| kl_loss | **false** | Dr.GRPO 底座 IF-04(与 ALFWorld 自洽);Search-R1/SEED 用 0.001,记脚注 |
| test/save_freq | 50 | val 2048 条约 5 分钟/次;25 会让 2650 步花 9 小时纯评测 |

### 3.3 并发(逐层,均有实测依据)

verl 的 `AgentLoopWorker` 对分到的 64 行(8 task × 8 副本)是
`asyncio.create_task` **一次性全发 + gather**,**worker 内零分批** ⇒ 串行等待
完全来自各 client 的并发信号量。配置写**全局值**,`_per_worker(cap,16)` 自动分摊。

| 层 | 配置(全局) | 每 worker 实得 | 需求 | 说明 |
|---|---|---|---|---|
| `agent_num_workers` | **16** | — | — | **不是必须 == batch**;条件是 `batch % workers == 0`(verl 按 worker 数对 interleave 后的 batch 连续切块)。取 16 而非 128:每 worker 一个独立检索 batcher,128 进程会让合批完全失效(实测 avg_batch=1、wait 47s) |
| `roles.researcher.max_concurrency` | **128** | 8 | 8 task | 旧值 64 ⇒ 每 worker 仅 4,**8 个 task 抢 4 个名额**,researcher 阶段墙钟翻倍 |
| `roles.narrator.max_concurrency` | **512** | 32 | 峰值 ≈ 8 | narration 每 episode 一次(~1415/步,11× researcher);余量充足 |
| `rollout_max_num_seqs` | 512 | — | ≤1024 总 | 4 replica × 512 = 2048,不限制 |
| `rollout_max_num_batched_tokens` | **32768** | — | — | SearchQA context 随轮次涨快(每次检索 +900 tok,第4轮 ~4050);8192 一批只装 2 个 prefill |
| `searchqa.search_concurrency` | **2**(在途批) | — | — | 服务端 ~700 q/s 即饱和;曾配 64 ⇒ 双臂 8192 query 在途,把服务端打到单发 13.7s |
| `searchqa.search_max_batch` | **512** | — | — | 对齐服务端内部 `retrieval_batch_size=512`(实测 1.95× 于 64) |

### 3.4 检索合批(必须,否则吞吐塌陷)

`envs/searchqa/adapter.py::_SearchBatcher`(进程级聚合)。两条硬约束:

1. **必须传 `return_scores=True`** —— 服务端在 `False` 且多 query 时返回**转置矩阵**
   (queries=2/topk=3 → len=3 的 [2,2,2]);单 query 恰好不受影响,故逐条实现从未暴露。
   代码内另有**形状硬校验**兜底。
2. 吞吐:单 query 单请求 **拉并发无效**(15.7 q/s 封顶);合批后实测 **600 q/s**
   (320 queries/160 并发)。每步 2765 query ⇒ 检索 <5 秒,占比 <1%。

### 3.5 narration 后端

`narration.backend: external`(与 researcher 同端点,但**独立 client**)。

- **动因**:narration 对每个 researcher episode 调一次 ⇒ 1415 次/步 ×
  (1141 prefill + 308 decode)= **1.62M prefill + 0.44M decode**,是单项最大的
  prefill 来源
- **A/B 实测**:engine 628.26 s/it → external 609.47 s/it;doc 质量无退化
  (词数 423→418、实验 4.2→5.16、干预 gap 同为 +3.9pp 且配对方向更干净)
- **语义变更须知**:engine 档是"policy 自述自己的轨迹"(历史用户裁定);
  external 档 researcher 读到的是"强模型眼中的轨迹"。**不引入新特权**
  (researcher 本就有权看轨迹全文),但视角变了
- ⚠ **不复用 researcher 的 client 实例**:两者调用量差 11×,共用并发池会互相饿死

---

## 4. 启动

```bash
# 冒烟(2 步,**不写 ckpt**)
ssh a800 'bash /fs/fast/u2021000902/wushang/workspace/TRACE-OPD-TERM/probe/launch_dual_sqa7b.sh smoke'

# 正式(2 epochs ≈ 2650 步)
ssh a800 'bash /fs/fast/u2021000902/wushang/workspace/TRACE-OPD-TERM/probe/launch_dual_sqa7b.sh formal'

# 看护(自愈;正式 run 才需要)
ssh a800 'W=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-TERM; \
  rm -f $W/runs/watchdog_classic_sqa7b.{stop,pid}; export TRACE_35B_KEY=token-abc123; \
  RUN_PREFIX=classic_sqa7b CFG_STEM=searchqa_7b RAY_TAG=sqa TOTAL_STEPS=2650 \
  setsid bash $W/probe/watchdog_dual.sh >/dev/null 2>&1 </dev/null &'

# 停看护 / 停某臂
ssh a800 'touch $W/runs/watchdog_classic_sqa7b.stop'
ssh a800 'bash $W/probe/kill_arm.sh classic_sqa7b_pgs classic_sqa7b_grpo'
```

发射脚本内置:`source env_pgs_sglang_a800.sh`(编译前置硬规则)、
export `TRACE_35B_KEY`/`SWANLAB_API_KEY`、端口按臂隔离(pgs 30000 / grpo 32000)、
错峰 20s、`setsid` 脱离终端。**冒烟额外加 `save_freq=100000` ⇒ 不写 ckpt**
(一次 ckpt = model 7.6G×4 + optim 9.5G×4 ≈ 68G/臂,双臂并发 136G,磁盘紧张时
会写坏:实测 `torch.save unexpected pos` + 同批 rollouts 全 0 字节)。

---

## 5. 观察

```bash
PY=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD/env_sglang_train/bin/python
W=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-TERM

$PY $W/probe/val_stat.py      <run>/rollouts/step_NNNNN_val   # 胜率×族×截断
$PY $W/probe/sqa_trunc.py     <run>/rollouts/step_NNNNN_val   # 截断撞墙归因
$PY $W/probe/steer_effect.py  <run>/rollouts/step_NNNNN       # steered vs bare 配对
$PY $W/probe/verify_grouping.py <run>/rollouts/step_NNNNN     # 每task 8副本/同doc/4+4
$PY $W/probe/fmt_check.py     <run>/rollouts/step_NNNNN_val   # 逐轮格式服从+轨迹取样

# 看板(本地):RUN_PREFIX=classic_sqa7b ./scope.sh
```

**读数纪律**:
- **console 的 step 指标行会被 Ray 转发吞掉**,不要 grep 日志找 `pg_loss`;
  读数一律取自 rollouts 产物或 swanlab
- 统计前**必须排除 0 字节文件**(`find … ! -size 0`):文件是"先创建后写入",
  半写状态会被 glob 计数成正常、再被 `json.load` 的 try/except 静默跳过,
  最终表现为"0 行数据"
- rollout 未跑完时的读数**系统性偏高**(先完成的是较易任务):实测早期 777 条
  时 EM 42.2%,全量 2048 收敛到 35.3%

---

## 6. 起点基线(文本协议,两轮独立复现)

| | PGS | GRPO |
|---|---|---|
| step0 val EM | 35.1% / 35.5% | 35.3% / 35.6% |
| 截断 / DEGENERATE / invalid | 0 / 0 / 0 | 0 / 0 / 0 |
| 有效组(组内既有赢又有输) | 44% | 42-44% |

族分解(GRPO step0):triviaqa 58.8% > popqa 39.3% > nq 37.4% > hotpotqa 31.0%
> 2wiki 17.9% > **musique 12.0%**(最难)。

**headroom**:Search-R1 论文 7B-instruct GRPO 训练后 **.385**;起点 35.3% ⇒
空间存在且多跳题(musique/2wiki 占 val 29%)余量大,**不会像 ALFWorld 那样
几十步饱和**(ALFWorld 基线 40 步即 93.6%,全对组吞掉 batch、有效组塌陷)。

---

## 7. 已知无害噪声

- `DataLoader worker (pid …) is killed by signal: Killed` —— 栈为
  `swanlab finish() → time.sleep() → torch signal_handling`,训练正常结束后
  worker 被清理、swanlab 收尾恰在 sleep 被信号处理器抛错;**不在训练路径上**
  (异常后仍完整打印 step 指标与 final validation)
- `logger.info` 的 `dynamic researcher ready` / `external narrator ready`
  **不进 launch.log**(Ray worker 日志转发),不能据此判断 client 未构造;
  判据看 `research/` 是否产出会话

## 8. 事故记录与硬规则(append-only)

- **2026-08-13 steering 守卫回归**:改 `agent_loop.py`(narrator 插入)时误删
  `if config.steering.enabled:`,GRPO 臂 worker init 全崩(基线臂吃
  `gate_adaptive=True` 默认值去构造 tracker)。⇒ **硬规则:凡动
  `src/trace_opd/train/agent_loop.py` 等双臂公共路径,冒烟必须双臂齐跑;
  只跑机制臂验不出基线臂回归**。详见 classic-bench-ops.md §13.13。
- **A800 残留清理**:训练主进程死后 sglang/ray 子进程可残留 ~50G/卡;
  按 `/proc/$p/exe` 前缀 `env_sglang_train/` 清理(勿按 cmdline pkill)。
