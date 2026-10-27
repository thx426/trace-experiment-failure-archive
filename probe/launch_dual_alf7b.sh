#!/bin/bash
# 双臂并行发射:PGS 机制臂(CVD 0-3)+ GRPO 基线臂(CVD 4-7)。
# 用法: launch_dual_alf7b.sh [smoke|formal]
MODE=${1:?usage: launch_dual_alf7b.sh smoke|formal}
W=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-TERM
# 编译前置入口(硬规则:勿绕过手拼 CUDA JIT/link 路径——冒烟v2 实测缺它则
# gcc 找不到 libcudart、Triton 回退 CPU)
source /fs/fast/u2021000902/wushang/workspace/TRACE-OPD-PGS/env_pgs_sglang_a800.sh
PYX=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD/env_sglang_train/bin/python
export ALFWORLD_DATA=/fs/fast/u2021000902/wushang/workspace/data/alfworld_data
export ALFWORLD_WORKER_PYTHON=/fs/fast/u2021000902/wushang/workspace/alfworld-venv/bin/python
export TRACE_35B_KEY=token-abc123
export SWANLAB_API_KEY=REDACTED
cd $W

fire() { local arm=$1 cvd=$2 raydir=$3 sglport=$4; shift 4
  local name="classic_alf7b_${arm}"
  local extra=()
  if [ "$MODE" = smoke ]; then
    name="${name}_smoke"
    extra=(trainer.total_training_steps=2)
  fi
  # SGLang 端口基址按臂隔离(冒烟v2 实测:单机双实例同默认基址 30000,
  # TCPStore 30701 竞态相撞;偏移 2000 隔离派生端口空间)
  extra+=("+actor_rollout_ref.rollout.engine_kwargs.sglang.port=${sglport}")
  setsid env CUDA_VISIBLE_DEVICES=$cvd RAY_TMPDIR=$raydir \
    $PYX scripts/launch_train.py \
      --config $W/configs/classic/alfworld_7b_${arm}.yaml \
      --run-name $name "${extra[@]}" \
    > $W/runs/${name}.launch.log 2>&1 < /dev/null &
  echo "FIRED $name cvd=$cvd sglang_port=$sglport pid=$!"
}

mkdir -p /tmp/ray_pgs /tmp/ray_grpo $W/runs
fire pgs  0,1,2,3 /tmp/ray_pgs  30000
sleep 20   # 错峰:让臂 A 的引擎端口扫描先落位
fire grpo 4,5,6,7 /tmp/ray_grpo 32000
echo "DUAL-FIRED mode=$MODE"
