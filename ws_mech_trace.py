"""把 α 的两个消费点(采样端/advantage 端)和停产逻辑完整拉出来。"""
import subprocess

def show(title, cmd, n=40):
    print(f"\n{'='*72}\n{title}\n{'='*72}")
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout
    print("\n".join(out.splitlines()[:n]) or "  (无)")

show("① α 的计算(cosine 调度)",
     "grep -rn --include=*.py -B3 -A18 'def steered_alpha' src/trace_opd/")

show("② advantage 端怎么用 α(pgs_asym_grpo)",
     "grep -rn --include=*.py -A30 'def compute_pgs_asym' src/trace_opd/ | head -45", 45)

show("③ 采样端停产的调用点(谁在问 retired)",
     "grep -rn --include=*.py '_steered_arm_retired\\|steered_retired' src/trace_opd/")

show("④ 停产后行角色怎么分配",
     "grep -rn --include=*.py -B6 -A14 'retired' src/trace_opd/train/agent_loop.py | head -45", 45)
