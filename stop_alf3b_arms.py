"""停止 ALFWorld 3B 双臂(classic_alf3b_pgs / classic_alf3b_grpo)。

纪律(踩过的坑):
  * **按 /proc 直读判定,不用 pgrep -f** —— 后者会匹配到执行清理的 ssh
    命令本身,导致自杀或计数偏高;
  * **三重保护**:必须命中双臂特征、且不在保护名单、且不是自己;
  * 先 SIGTERM 给 verl 落盘的机会,等待后再看残留;
  * 干跑模式默认开,`--go` 才真杀。

用法(A800):
  python3 stop_alf3b_arms.py          # 干跑,只列出将被杀的
  python3 stop_alf3b_arms.py --go     # 真停
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

# 命中任一即视为双臂进程
MATCH = ("classic_alf3b_pgs", "classic_alf3b_grpo",
         "TRACE-OPD-TERM", "env_sglang_train")
# 命中任一即**豁免**(即使匹配上也不杀)
PROTECT = ("pgs_microscope_server", "webshop_service", "stop_alf3b_arms",
           "rollout_viewer", "sshd", "swan_read", "val_all")


def cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (OSError, PermissionError):
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", "replace").strip()


def environ_cvd(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except (OSError, PermissionError):
        return ""
    for kv in raw.split(b"\0"):
        if kv.startswith(b"CUDA_VISIBLE_DEVICES="):
            return kv.split(b"=", 1)[1].decode()
    return ""


#: 训练栈的解释器 —— ★ ray 会把 worker 的 cmdline 改写成 `ray::TaskName`,
#: 路径信息全丢,只靠 cmdline 关键词会漏掉一批僵尸 worker(2026-08-16 实测
#: 漏了 9 个)。exe 路径不受改写影响,是更可靠的归属判据。
TRAIN_EXE = "/env_sglang_train/bin/python"


def exe_of(pid: int) -> str:
    try:
        return os.readlink(f"/proc/{pid}/exe")
    except (OSError, PermissionError):
        return ""


def scan() -> list[tuple[int, str, str]]:
    me, parent = os.getpid(), os.getppid()
    out = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in (me, parent, 1):
            continue
        cmd = cmdline(pid)
        if not cmd:
            continue
        # 命中条件:cmdline 关键词 **或** 训练栈解释器(覆盖 ray 改写的 worker)
        hit = any(m in cmd for m in MATCH) or TRAIN_EXE in exe_of(pid)
        if not hit:
            continue
        if any(p in cmd for p in PROTECT):
            continue
        out.append((pid, environ_cvd(pid), cmd[:110]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="真的发信号(默认干跑)")
    ap.add_argument("--wait", type=float, default=45.0)
    a = ap.parse_args()

    hits = scan()
    print(f"[scan] 命中 {len(hits)} 个进程")
    cvds: dict[str, int] = {}
    for _pid, cvd, _cmd in hits:
        cvds[cvd or "(无)"] = cvds.get(cvd or "(无)", 0) + 1
    print(f"[scan] CUDA_VISIBLE_DEVICES 分布: {cvds}")
    for pid, cvd, cmd in hits[:8]:
        print(f"   {pid:>8} cvd={cvd or '-':<8} {cmd}")
    if len(hits) > 8:
        print(f"   … 另 {len(hits) - 8} 个")

    # 保护名单命中情况(必须能看到它们被豁免,否则说明判据写错了)
    prot = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            c = cmdline(int(entry.name))
            if c and any(p in c for p in PROTECT) and "sshd" not in c:
                prot.append((int(entry.name), c[:80]))
    print(f"[prot] 豁免 {len(prot)} 个: {[p for p, _ in prot][:6]}")

    if not a.go:
        print("\n[dry-run] 未发信号。确认无误后加 --go")
        return 0

    print(f"\n[kill] SIGTERM × {len(hits)}")
    for pid, _cvd, _cmd in hits:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + a.wait
    while time.time() < deadline:
        time.sleep(3)
        left = scan()
        print(f"  剩余 {len(left)}", flush=True)
        if not left:
            break
    left = scan()
    if left:
        print(f"[kill] SIGKILL 残留 {len(left)}")
        for pid, _c, _m in left:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(3)
    print(f"\n[done] 最终残留 {len(scan())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
