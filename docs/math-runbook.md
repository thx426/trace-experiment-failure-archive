# Math(DAPO-17k × 六竞赛源)× Qwen2.5-7B 双臂 RUNBOOK

> 单一真源(2026-08-13 冒烟全绿后用户确认为默认)。配置真源
> `configs/classic/math_7b_{pgs,grpo}.yaml`,运行脚本
> `probe/launch_dual_math7b.sh`。过程记录见 classic-bench-ops.md §13.15-13.18。

## 1. 前置依赖

- **162 的 35B researcher 双端点**(8880/8881,带 `TRACE_35B_KEY` 鉴权)——
  仅 PGS 臂需要;narration 同端点(external 档)。验证:
  `curl -H "Authorization: REDACTED $KEY" http://HOST_IP:8880/v1/models` → 200。
- **无检索/无容器依赖**:math 环境是本机 subprocess(`envs/pytool.py`),
  不碰 162 retriever。`env_sglang_train` 需 `math-verify`(antlr4_9_3 档)与
  `sympy`(已装:1.14)。
- A800 入口:`source TRACE-OPD-PGS/env_pgs_sglang_a800.sh`;
  **⚠ `export PYTHONPATH=$W/src` 必须写在 source 之后**(入口会覆写到 PGS 树,
  否则 `-m trace_opd...` 静默跑错树)。

## 2. 数据契约

`/fs/fast/u2021000902/wushang/data/classic_math/`:
- `train.parquet` 17,255(DAPO-17k 按题面 sha1 去重;2 epochs ≈ 270 步)
- `valid.parquet` 1,304 六桶:aime24/25/26 各30、amc23 40、math500 500、
  olympiad 674;`data_source=math/<bench>` 出 by-family 曲线
- schema 原生带 `agent_name='trace_opd'` + `reward_model.style='env'`
  (verl trainer 两必需字段,builder 写入)
- **换 system prompt 无需重烘**:trainer/evalm 均走
  `adapter.executor_scaffold()` 运行时组装(agent_loop.py:517),parquet
  prompt 列是装饰性的

## 3. 协议(2026-08-13 用户裁定:文本 TIR)

对照 ToRL(2503.23383)/SimpleTIR(2509.02479,同基座)/ReTool(2504.11536):
每轮 = 推理 + 二选一:``` ```python ``` 围栏完整脚本(print 中间量,变量不
跨块)→ 环境回 ``` ```output ``` ;或无代码轮的 `\boxed{}` 终答。one-shot
内嵌 system。**代码>boxed**(同轮并存执行代码——防幻觉输出);boxed 抽取
花括号平衡扫描;错误全文回喂。**协议效应:同集同判分 FC 55.2% → text
76.8%(math500 全量,+21.6pp;还原模型原生水平)**。

## 4. 启动

```bash
ssh a800
W=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-TERM
bash $W/probe/launch_dual_math7b.sh formal     # smoke 档:2 步不写 ckpt
# watchdog(参数化通用件):
setsid env RUN_PREFIX=classic_math7b CFG_STEM=math_7b RAY_TAG=math TOTAL_STEPS=270 \
  bash $W/probe/watchdog_dual.sh > /dev/null 2>&1 &
```
PGS=CVD0-3/port30000,GRPO=CVD4-7/port32000;错峰 20s;ray tmpdir
`/tmp/ray_math_{pgs,grpo}`。

## 5. 观察工具与读数纪律

- `probe/math_smoke_stat.py <step目录> [--val]`:分桶 EM/用码率/void/干预配对
- `probe/math_doc_leak.py <run目录>`:答案入 doc 率 + 分层 gap(**数学域必报**)
- `probe/math_traj_dump.py <step目录> N [won|lost|steered|code]`:可读轨迹
- 纪律:统计排 0 字节;未跑完读数偏高;steering 读数只认副本齐 8 的 task;
  **答案通道 vs 策略通道要随步数持续分层**(答案入 doc 率 90-95% 是数学域
  常态:35B oracle 实验自验答案,`gt_injected=False` 已全量核验)

## 6. 起点基线(冒烟实测 2026-08-13)

- val 1304:总 **54.8%**(math500 78.0/amc 60/olympiad 43/aime24 20/
  aime26 13/aime25 7);evalm↔训练双路径差 1.2pp 交叉验证过
- 训练态(temp1):won ~32-37%、**有效组 58%**、用码率 42-43%、零截断
- step1=哨兵(干预 100% 为 0,gap −2.1pp 零对照);step2 门落位
  (θ=1.341/开门率 15.6%/强度 0.43):**steered 41.0 vs bare 33.6,
  +7.4pp(40W/69T/19N,p≈0.008)**;干预 token 中位 221
- researcher:实验 4.45/会话、final_success 61.9%、doc 中位 ~411 词
- 墙钟(双臂共跑):GRPO 393s/步、PGS 860-1532s/步 ⇒ 270 步全程
  GRPO ≈1.5 天、PGS ≈3-4 天

## 7. 已知无害噪声与遗留

- **swanlab atexit 噪声**:退出时 "Exception ignored in atexit callback …
  DataLoader worker killed" —— 上传轮询撞 teardown,产物与指标完整,非故障
- **PGS 臂末步强制写 ckpt**(save_freq 不拦;机制门状态保全)——冒烟后
  记得清 `runs/*_smoke/checkpoints`(一份 68G)
- 开门率首步落点 15.6%(target 0.05,告警带 [0.05,0.25] 内)——正式跑观察
  是否回落;asym 退火 180 步 cosine 是克隆压力防线
