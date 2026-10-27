#!/bin/bash
# SearchQA 双臂并行发射:PGS 机制臂(CVD 0-3)+ GRPO 基线臂(CVD 4-7)。
# 用法: launch_dual_sqa7b.sh [smoke|formal]
MODE=${1:?usage: launch_dual_sqa7b.sh smoke|formal}
W=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-TERM
# 编译前置入口(硬规则:勿绕过手拼 CUDA JIT/link 路径)
source /fs/fast/u2021000902/wushang/workspace/TRACE-OPD-PGS/env_pgs_sglang_a800.sh
PYX=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD/env_sglang_train/bin/python
export TRACE_35B_KEY=token-abc123
export SWANLAB_API_KEY=REDACTED
cd $W

fire() { local arm=$1 cvd=$2 raydir=$3 sglport=$4
  local name="classic_sqa7b_${arm}"
  local extra=()
  if [ "$MODE" = smoke ]; then
    name="${name}_smoke"
    # save_freq 调大到不可达 ⇒ 冒烟**不写 checkpoint**。
    # 2026-08-13:一次 ckpt = model 7.6G×4 + optim 9.5G×4 ≈ 68G/臂,双臂并发
    # 136G;A800 磁盘紧张时会写坏(实测 grpo 报 torch.save
    # "unexpected pos 8606802624 vs 8606802524",同批 step2 rollouts 全 1024
    # 个文件 0 字节)。冒烟只需 rollout 产物(约 78M),不需要 ckpt。
    extra=(trainer.total_training_steps=2 trainer.save_freq=100000)
  fi
  # SGLang 端口基址按臂隔离(ALFWorld 冒烟实证:同默认基址会 TCPStore 竞态相撞)
  extra+=("+actor_rollout_ref.rollout.engine_kwargs.sglang.port=${sglport}")
  setsid env CUDA_VISIBLE_DEVICES=$cvd RAY_TMPDIR=$raydir \
    $PYX scripts/launch_train.py \
      --config $W/configs/classic/searchqa_7b_${arm}.yaml \
      --run-name $name "${extra[@]}" \
    >> $W/runs/${name}.launch.log 2>&1 < /dev/null &
  echo "FIRED $name cvd=$cvd sglang_port=$sglport pid=$!"
}

mkdir -p /tmp/ray_sqa_pgs /tmp/ray_sqa_grpo $W/runs
fire pgs  0,1,2,3 /tmp/ray_sqa_pgs  30000
sleep 20   # 错峰:让臂 A 的引擎端口扫描先落位
fire grpo 4,5,6,7 /tmp/ray_sqa_grpo 32000
echo "DUAL-FIRED mode=$MODE (SearchQA)"
