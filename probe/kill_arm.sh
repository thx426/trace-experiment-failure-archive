#!/usr/bin/env bash
# 停指定 run(按 experiment_name 精确匹配 cmdline),并清理其 ray/alfworld 子进程。
# 用法: kill_arm.sh <experiment_name> [<experiment_name> ...]
# 纪律:用 python 读 /proc 避免 pgrep/pkill 自匹配(本脚本自身 cmdline 含关键词)。
set -u
SELF_PID=$$
PY=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD/env_sglang_train/bin/python

for NAME in "$@"; do
  echo "=== 停 $NAME ==="
  $PY - "$NAME" "$SELF_PID" <<'PYEOF'
import os, sys, signal, time
name, self_pid = sys.argv[1], int(sys.argv[2])
def cmdline(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode("utf8", "replace")
    except Exception:
        return ""
def match():
    out = []
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        pid = int(p)
        if pid in (self_pid, os.getpid()) or pid == os.getppid():
            continue
        c = cmdline(pid)
        if not c or "kill_arm.sh" in c or "/proc/" in c:
            continue
        if name in c:
            out.append((pid, c[:150]))
    return out
hits = match()
print(f"  匹配 {len(hits)} 个进程")
for pid, c in hits[:8]:
    print(f"    {pid}: {c}")
for pid, _ in hits:
    try: os.kill(pid, signal.SIGTERM)
    except Exception: pass
time.sleep(8)
left = match()
for pid, _ in left:
    try: os.kill(pid, signal.SIGKILL)
    except Exception: pass
time.sleep(3)
print(f"  SIGTERM 后残留 {len(left)} → SIGKILL;最终残留 {len(match())}")
PYEOF
done

echo "=== 清理孤儿 alfworld worker(父进程已死)==="
$PY - <<'PYEOF'
import os, signal, time
def info(pid):
    try:
        c = open(f"/proc/{pid}/cmdline","rb").read().replace(b"\0",b" ").decode("utf8","replace")
        ppid = 0
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("PPid:"): ppid = int(line.split()[1])
        return c, ppid
    except Exception:
        return "", -1
orphans = []
for p in os.listdir("/proc"):
    if not p.isdigit(): continue
    pid = int(p)
    c, ppid = info(pid)
    if "alfworld" in c and "worker" in c and ppid == 1:
        orphans.append(pid)
print(f"  孤儿 alfworld worker: {len(orphans)}")
for pid in orphans:
    try: os.kill(pid, signal.SIGKILL)
    except Exception: pass
PYEOF

echo "=== 结果 ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
uptime
