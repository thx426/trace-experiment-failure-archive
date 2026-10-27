#!/bin/bash
# Math 双臂并行发射:PGS 机制臂(CVD 0-3)+ GRPO 基线臂(CVD 4-7)。
# 用法: launch_dual_math7b.sh [smoke|formal]   (照抄 launch_dual_sqa7b.sh 定式)
MODE=${1:?usage: launch_dual_math7b.sh smoke|formal}
W=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-TERM
source /fs/fast/u2021000902/wushang/workspace/TRACE-OPD-PGS/env_pgs_sglang_a800.sh
export PYTHONPATH=$W/src:$W/verl   # ★ 必须在 source 之后(入口会覆写到 PGS 树)
PYX=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD/env_sglang_train/bin/python
export TRACE_35B_KEY=token-abc123
export SWANLAB_API_KEY=REDACTED
cd $W

fire() { local arm=$1 cvd=$2 raydir=$3 sglport=$4
  local name="classic_math7b_${arm}"
  local extra=()
  if [ "$MODE" = smoke ]; then
    name="${name}_smoke"
    # 冒烟不写 ckpt(磁盘紧张时写坏的实证见 sqa 卡片)
    extra=(trainer.total_training_steps=2 trainer.save_freq=100000)
  fi
  # SGLang 端口基址按臂隔离(同默认基址会 TCPStore 竞态相撞,ALFWorld 实证)
  extra+=("+actor_rollout_ref.rollout.engine_kwargs.sglang.port=${sglport}")
  setsid env CUDA_VISIBLE_DEVICES=$cvd RAY_TMPDIR=$raydir \
    $PYX scripts/launch_train.py \
      --config $W/configs/classic/math_7b_${arm}.yaml \
      --run-name $name "${extra[@]}" \
    >> $W/runs/${name}.launch.log 2>&1 < /dev/null &
  echo "FIRED $name cvd=$cvd sglang_port=$sglport pid=$!"
}

mkdir -p /tmp/ray_math_pgs /tmp/ray_math_grpo $W/runs
fire pgs  0,1,2,3 /tmp/ray_math_pgs  30000
sleep 20
fire grpo 4,5,6,7 /tmp/ray_math_grpo 32000
echo "DUAL-FIRED mode=$MODE (Math)"
