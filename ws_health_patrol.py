"""WebShop 双臂的健康巡检(只读,可反复跑;供长期看护用)。

判据取自本轮踩过的坑,每条都对应一个真实故障模式:
  H1 进程存活   —— 按 exe 路径判,不用 pgrep -f(ray 会把 worker 的 cmdline
                    改写成 `ray::TaskName`,关键词匹配会漏,2026-08-17 实测漏 9 个)
  H2 GPU 在算   —— 只看 memory.used 会把"僵尸占显存"误判为健康,必须看
                    utilization;进度停滞而显存还在 = 驱动楔死的指纹
  H3 步进推进   —— 与上次巡检比 rollout 目录是否增长(存进度文件到 --state)
  H4 环境服务   —— 逐分片 /meta:数量、num_goals 一致、option_fix 一致、
                    池未干涸(free=0 且 busy 不变 = 泄漏)、reaped 是否在涨
  H5 读数健康   —— 最近一步的 ENV_ERROR / invalid_format / truncated 比例
  H6 磁盘       —— ckpt 只留 1 个是用户硬要求,存盘峰值要 2×

退出码:0 全绿;1 有 WARN;2 有 FATAL(需要人介入)。

用法(A800,仓库根):
  python3 ws_health_patrol.py --state /tmp/ws_patrol.json
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import time
from pathlib import Path

RUNS = ("classic_ws3b_pgs", "classic_ws3b_grpo")
ARM_CVD = {"classic_ws3b_pgs": "4,5,6,7", "classic_ws3b_grpo": "0,1,2,3"}
TRAIN_EXE = "/env_sglang_train/bin/python"
#: ★ 端点**从配置读**,不硬编码 —— 分片数会随实验规模调整,硬编码的范围
#: 会在配置变更后报假 FATAL(2026-08-17 踩过:配置从 48 改回 32 分片后,
#: 巡检仍去查 8342-8357 并判"16 个分片不可达")。
CFG = "configs/classic/webshop_3b_pgs.yaml"

FATAL, WARN, OK = "FATAL", "WARN", "OK"
findings: list[tuple[str, str, str]] = []


def note(level: str, tag: str, msg: str) -> None:
    findings.append((level, tag, msg))


def sh(cmd: str, timeout: int = 30) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=timeout).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def h1_processes() -> None:
    n = 0
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            if TRAIN_EXE in Path(f"/proc/{d.name}/exe").resolve().as_posix():
                n += 1
        except (OSError, PermissionError):
            pass
    note(OK if n > 50 else FATAL, "H1 进程",
         f"训练栈进程 {n} 个(按 exe 路径判;<50 视为异常)")


def h2_gpu(stalled: bool) -> None:
    """★ 判据必须是「占显存零利用 **且** 进度停滞」。

    只看 util==0 会把 rollout 阶段的正常空窗(GPU 在等环境返回,显存被模型
    常驻占着)报成故障 —— 2026-08-17 实测 20 次巡检里 8 次 WARN 全是这个
    误报,而 H3 进度一路在推进。真正的僵尸/驱动楔死指纹是两者同时成立。
    """
    out = sh("nvidia-smi --query-gpu=index,memory.used,utilization.gpu "
             "--format=csv,noheader,nounits")
    busy_idle = []
    for line in out.splitlines():
        try:
            i, mem, util = (x.strip() for x in line.split(","))
            if int(mem) > 5000 and int(util) < 5:
                busy_idle.append(f"GPU{i}({int(mem)//1024}G/{util}%)")
        except ValueError:
            pass
    if busy_idle and stalled:
        note(FATAL, "H2 GPU",
             f"占显存零利用 **且进度停滞**: {busy_idle} —— 僵尸/驱动楔死的"
             f"指纹,按 exe 路径清进程后重启")
    elif busy_idle:
        note(OK, "H2 GPU",
             f"{len(busy_idle)} 卡瞬时空窗(rollout 等环境,进度正常推进): "
             f"{busy_idle[:4]}")
    else:
        note(OK, "H2 GPU", "全部在算")


def h3_progress(state: dict) -> dict:
    new_state = {}
    for r in RUNS:
        dirs = sorted(p for p in glob.glob(f"runs/{r}/rollouts/step_*")
                      if not p.endswith("_val"))
        cur = len(dirs)
        new_state[r] = cur
        prev = state.get(r)
        last = dirs[-1].split("step_")[-1] if dirs else "-"
        # ★ 跑完 total_steps 后不再推进是**正常终止**,不是停滞
        done = Path(f"runs/{r}.launch.log").exists() and "100%|" in sh(
            f"grep -oE 'Training Progress: *100%\\|' runs/{r}.launch.log | tail -1")
        if done:
            note(OK, "H3 进度", f"{r}: **已完成** {cur} 步(不再推进属正常)")
        elif prev is None:
            note(OK, "H3 进度", f"{r}: {cur} 步(首次巡检,下次可比)")
        elif cur > prev:
            note(OK, "H3 进度", f"{r}: {prev}→{cur} 步(最新 {last})")
        else:
            note(WARN, "H3 进度",
                 f"{r}: 仍是 {cur} 步(最新 {last})—— 单步可达 5-10 分钟,"
                 f"连续两次不动才算停滞")
    return new_state


def _endpoints() -> list[str]:
    """从生效配置里取分片地址(与训练进程用的是同一份真源)。"""
    import yaml
    cfg = yaml.safe_load(Path(CFG).read_text(encoding="utf-8"))
    url = str((cfg.get("webshop") or {}).get("service_url", ""))
    return [u.strip().rstrip("/") for u in url.split(",") if u.strip()]


def h4_env() -> None:
    metas, bad = {}, []
    for p in _endpoints():
        o = sh(f"curl -s --max-time 4 {p}/meta", timeout=8)
        if not o:                      # ★ 重试一次再判:服务繁忙时单次 curl
            time.sleep(1)              #   会瞬时超时,2026-08-17 曾据此误报
            o = sh(f"curl -s --max-time 8 {p}/meta", timeout=12)
        if not o:
            bad.append(p)
            continue
        try:
            metas[p] = json.loads(o)
        except Exception:  # noqa: BLE001
            bad.append(p)
    if bad:
        # 少量分片失联 ⇒ 客户端有 429 退避与重试,训练不会错,只会慢 ⇒ WARN;
        # 大面积失联才是真故障。
        lvl = FATAL if len(bad) > max(1, len(_endpoints()) * 0.1) else WARN
        note(lvl, "H4 环境",
             f"{len(bad)}/{len(_endpoints())} 个分片两次探测均不可达: {bad[:6]}"
             f"{' —— 大面积失联,需介入' if lvl == FATAL else ' (少量,客户端会退避重试)'}")
        if lvl == FATAL:
            return
    ng = {m.get("num_goals") for m in metas.values()}
    fx = {bool(m.get("option_fix")) for m in metas.values()}
    slots = sum(int(m.get("pool", 0)) for m in metas.values())
    busy = sum(int(m.get("busy", 0)) for m in metas.values())
    reap = sum(int(m.get("reaped", 0)) for m in metas.values())
    note(OK if len(ng) == 1 else FATAL, "H4 环境",
         f"{len(metas)} 分片 / 槽位 {slots} / num_goals {ng} / option_fix {fx}")
    if len(fx) != 1 or not fx.pop():
        note(FATAL, "H4 环境", "option_fix 不一致或未开 —— reward 口径分歧")
    dry = [p for p, m in metas.items()
           if int(m.get("free", 0)) == 0 and int(m.get("busy", 0)) > 0]
    note(WARN if len(dry) > len(metas) * 0.5 else OK, "H4 池",
         f"busy={busy}/{slots} ({busy/max(1,slots):.0%});干涸分片 {len(dry)};"
         f"reaped 累计 {reap}{' ← TTL 在回收僵尸 sid,查客户端超时' if reap else ''}")


def h5_readings() -> None:
    for r in RUNS:
        dirs = sorted(p for p in glob.glob(f"runs/{r}/rollouts/step_*")
                      if not p.endswith("_val"))
        if not dirs:
            note(WARN, "H5 读数", f"{r}: 尚无训练步产出")
            continue
        recs = []
        for f in glob.glob(f"{dirs[-1]}/*.json"):
            try:
                recs.append(json.loads(Path(f).read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                pass
        if not recs:
            continue
        n = len(recs)
        err = sum(1 for x in recs if x.get("outcome") == "ENV_ERROR")
        att = [x for x in recs if x.get("outcome") != "ENV_ERROR"]
        won = sum(1 for x in att if x.get("won"))
        sc = [float((x.get("trajectory") or {}).get("score") or 0) for x in att]
        invf = sum(int(x.get("invalid_format") or 0) for x in att)
        tr = sum(1 for x in att if x.get("truncated"))
        lvl = FATAL if err > n * 0.05 else (WARN if err else OK)
        note(lvl, "H5 读数",
             f"{r} {Path(dirs[-1]).name}: n={n} ENV_ERROR={err} "
             f"Score={sum(sc)/max(1,len(sc))*100:.1f} Succ={won/max(1,len(att)):.2%} "
             f"invalid={invf} trunc={tr}")


def h6_disk() -> None:
    out = sh("df -h /fs/fast | tail -1")
    for r in RUNS:
        cks = sorted(glob.glob(f"runs/{r}/checkpoints/global_step_*"))
        big = [c for c in cks
               if int(sh(f"du -sm {c} 2>/dev/null | cut -f1") or 0) > 1000]
        if len(big) > 1:
            note(WARN, "H6 磁盘",
                 f"{r}: {len(big)} 个实体 ckpt(用户硬要求只留 1 个): {big}")
    note(OK, "H6 磁盘", f"/fs/fast: {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="/tmp/ws_patrol.json")
    a = ap.parse_args()
    state = {}
    sp = Path(a.state)
    if sp.exists():
        try:
            state = json.loads(sp.read_text())
        except Exception:  # noqa: BLE001
            pass

    print(f"=== WebShop 双臂巡检 {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    h1_processes()
    new_state = h3_progress(state)          # 先算进度,H2 要用它做联合判据
    stalled = bool(state) and all(new_state.get(r, 0) <= state.get(r, -1)
                                  for r in RUNS)
    h2_gpu(stalled)
    h4_env()
    h5_readings()
    h6_disk()

    sp.write_text(json.dumps(new_state))
    rank = {FATAL: 0, WARN: 1, OK: 2}
    for lvl, tag, msg in sorted(findings, key=lambda x: rank[x[0]]):
        print(f"  [{lvl:5}] {tag}: {msg}")
    if any(l == FATAL for l, _, _ in findings):
        print("\n判决: ✗ FATAL —— 需要人介入")
        return 2
    if any(l == WARN for l, _, _ in findings):
        print("\n判决: ⚠ 有 WARN —— 观察,连续多次同一 WARN 才处理")
        return 1
    print("\n判决: ★ 全绿")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
