#!/usr/bin/env python3
"""长期健康巡检:把 2026-08-14 两起事故的教训做成自动检测。

只在**异常**时打印(静默 = 健康),供 Monitor 挂载。
用法: health_patrol.py [--once] [--interval 600]
"""
import os, sys, time, subprocess, json

W = "/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-TERM"
# run_name -> (首张 GPU, ray_tag)
ARMS = {
    "classic_sqa7b_pgs": (0, "sqa"),
    "classic_math7b_pgs": (4, "math"),
}
WATCHDOGS = ["classic_sqa7b", "classic_math7b"]


def proc_count(run_name: str) -> int:
    n = 0
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            c = open(f"/proc/{p}/cmdline", "rb").read().replace(b"\0", b" ").decode("utf8", "replace")
        except Exception:
            continue
        if f"--run-name {run_name} " in c + " ":
            n += 1
    return n


def gpu_used_max(idx: int, samples: int = 3, gap: int = 25) -> int:
    """取多次采样的**最大值**。verl 在训练阶段让 rollout 引擎 sleep 释放显存
    (实测同一卡在 24.7G 与 61G 之间波动),单次瞬时值会造成缩水误报。"""
    vals = []
    for i in range(samples):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(idx)],
                capture_output=True, text=True, timeout=20)
            vals.append(int(out.stdout.strip().splitlines()[0]))
        except Exception:
            pass
        if i < samples - 1:
            time.sleep(gap)
    return max(vals) if vals else -1


def distinct_runners(run_name: str) -> int:
    """近期日志里不同的 TaskRunnerV1 pid 数 —— >1 说明多实例并存(事故指纹)。"""
    log = f"{W}/runs/{run_name}.launch.log"
    if not os.path.exists(log):
        return 0
    try:
        with open(log, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 2_000_000))
            tail = f.read().decode("utf8", "replace").replace("\r", "\n")
    except Exception:
        return 0
    import re, datetime
    # launch.log 是 append 模式,累积历次启动的 pid ⇒ 必须按日志时间戳窗口过滤,
    # 否则重启过的 run 必然误报多实例。取最近 10 分钟内的行。
    now = datetime.datetime.now()
    pids = set()
    for line in tail.splitlines()[-4000:]:
        m = re.search(r"TaskRunnerV1 pid=(\d+)", line)
        if not m:
            continue
        t = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
        if t:
            try:
                ts = datetime.datetime.strptime(t.group(1), "%Y-%m-%d %H:%M:%S")
                if (now - ts).total_seconds() > 600:
                    continue
            except Exception:
                pass
        else:
            continue          # 无时间戳的行不计(无法判断新旧)
        pids.add(m.group(1))
    return len(pids)


def http_code(url: str, post_json=None, auth=None) -> str:
    cmd = ["curl", "-s", "-m", "10", "-o", "/dev/null", "-w", "%{http_code}"]
    if auth:
        cmd += ["-H", f"Authorization: REDACTED {auth}"]
    if post_json is not None:
        cmd += ["-X", "POST", "-H", "Content-Type: application/json", "-d", json.dumps(post_json)]
    cmd.append(url)
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return "000"


def ckpt_incomplete(run_name: str):
    """★ 零 IO 的磁盘问题检测(2026-08-14 用户裁定:不得用 dd 写盘探测,
    那会抢训练的 IO 影响实验)。

    改为**事后验证**:配额耗尽时 torch.save 会留下不完整的 ckpt 目录
    (实测 8/19 文件、42G/86G),而 resume_mode=auto 会优先加载它 ⇒ 直接检查
    最新 ckpt 的文件数即可,既零开销又命中真正要防的后果。"""
    ck_root = f"{W}/runs/{run_name}/checkpoints"
    if not os.path.isdir(ck_root):
        return None
    steps = []
    for d in os.listdir(ck_root):
        if d.startswith("global_step_"):
            try: steps.append((int(d.rsplit("_", 1)[1]), d))
            except Exception: pass
    if not steps:
        return None
    _, latest = max(steps)
    path = os.path.join(ck_root, latest)
    # 目录刚建立时文件还在写,给 10 分钟宽限期再判
    if time.time() - os.path.getmtime(path) < 600:
        return None
    nfiles = sum(len(fs) for _, _, fs in os.walk(path))
    if nfiles < 15:        # 完整的 7B actor ckpt 实测 19 个文件
        return f"{latest} 仅 {nfiles} 个文件(完整应 ~19)—— 疑似写坏,resume 会加载到它"
    return None


def check_once():
    issues = []
    for run, (g, _tag) in ARMS.items():
        n = proc_count(run)
        if n == 0:
            issues.append(f"[{run}] 主进程缺失(watchdog 应在 5 分钟内接管;连续两次告警=自愈失败)")
            continue
        # ★ 事故检测一:引擎显存缩水(残留实例抢显存致新引擎静默降级)
        used = gpu_used_max(g)
        if 1000 < used < 30000:
            issues.append(f"[{run}] ⚠ GPU{g} 引擎显存峰值仅 {used}MiB(3 次采样最大值 <30G)"
                          f"—— 疑似缩水启动,查残留实例;缩水不会自愈,须清干净后重启")
        # ★ 事故检测二:多实例并存
        d = distinct_runners(run)
        if d > 1:
            issues.append(f"[{run}] ⚠ 近期日志有 {d} 个不同 TaskRunnerV1 pid —— 多实例抢卡,"
                          f"按 RAY_TMPDIR 清旧实例")
    # ★ 事故检测三:ckpt 完整性(零 IO;替代原 dd 写盘探测)
    for run in ARMS:
        bad = ckpt_incomplete(run)
        if bad:
            issues.append(f"[{run}] ⚠ {bad} —— 立即删除该 ckpt 并查磁盘配额")
    # 依赖服务
    if http_code("http://HOST_IP:8000/retrieve",
                 post_json={"queries": ["ping"], "topk": 1, "return_scores": True}) != "200":
        issues.append("⚠ 162 retriever 不可达 —— SearchQA 会 fail loud 崩溃")
    if sum(1 for p in (8880, 8881)
           if http_code(f"http://HOST_IP:{p}/v1/models", auth="token-abc123") == "200") == 0:
        issues.append("⚠ 162 researcher 双端点全不可达 —— 机制臂会静默降级成 GRPO")
    # watchdog 在岗
    for t in WATCHDOGS:
        try:
            pid = open(f"{W}/runs/watchdog_{t}.pid").read().strip()
            alive = os.path.isdir(f"/proc/{pid}")
        except Exception:
            alive = False
        if not alive:
            issues.append(f"⚠ watchdog_{t} 不在岗 —— 该实验失去自愈保护")
    if issues:
        print(f"[{time.strftime('%m-%d %H:%M')}] 巡检异常:")
        for i in issues:
            print(f"    {i}")
        sys.stdout.flush()
    return len(issues)


if __name__ == "__main__":
    once = "--once" in sys.argv
    interval = 600
    if "--interval" in sys.argv:
        interval = int(sys.argv[sys.argv.index("--interval") + 1])
    if once:
        n = check_once()
        if n == 0:
            print("巡检通过:全部健康")
        sys.exit(0)
    while True:
        try:
            check_once()
        except Exception as e:
            print(f"[巡检自身异常] {e}")
            sys.stdout.flush()
        time.sleep(interval)
