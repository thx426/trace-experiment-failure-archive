"""Recompute the objective audit from preserved sources and local run artifacts.

Read-only with respect to training code/configurations/logs. Writes audit outputs
beside this script; first-run source snapshots are never silently overwritten.
No GPU execution or claims of historical runtime completion.
"""
from pathlib import Path
import csv
import difflib
import hashlib
import json
import math
import re
import shlex
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def write(name, data):
    (HERE / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


sources = [
    "scripts/launch_train.py", "src/trace_opd/common/config.py",
    "src/trace_opd/train/agent_loop.py", "src/trace_opd/core/episode.py",
    "src/trace_opd/steering/hook.py", "src/trace_opd/steering/kernel.py",
    "src/trace_opd/steering/sps.py", "src/trace_opd/steering/arms.py",
    "verl/verl/experimental/agent_loop/agent_loop.py",
    "verl/verl/trainer/ppo/core_algos.py",
    "verl/verl/trainer/ppo/v1/trainer_base.py",
    "verl/verl/trainer/ppo/v1/agent_loop_tq.py",
    "verl/verl/workers/utils/losses.py",
    "verl/verl/trainer/config/actor/actor.yaml",
    "src/trace_opd/envs/alfworld/adapter.py",
    "src/trace_opd/envs/webshop/adapter.py",
    "src/trace_opd/envs/searchqa/adapter.py",
]
source_records = []
source_diffs = []
for relative in sources:
    src = ROOT / relative
    dst = HERE / "current_source_snapshot" / relative
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    record = digest(dst)
    record.update(original_path=str(src), matches_current=dst.read_bytes() == src.read_bytes())
    source_records.append(record)
    if not record["matches_current"]:
        source_diffs.extend(difflib.unified_diff(
            dst.read_text().splitlines(True), src.read_text().splitlines(True),
            fromfile=str(dst), tofile=str(src)))
(HERE / "source-drift.patch").write_text("".join(source_diffs))

legacy = []
for row in json.loads((HERE / "legacy-source-register.json").read_text()):
    path = HERE / ("legacy_" + row["commit"][:8]) / row["path"]
    data = path.read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    legacy.append({**row, **digest(path), "actual_git_blob_sha1": blob,
                   "blob_matches": blob == row["git_blob_sha1"]})

manifests, invalid = [], []
for path in sorted((ROOT / "runs").glob("*/launch_manifest*.json")):
    try:
        d = json.loads(path.read_text())
    except (ValueError, OSError) as exc:
        invalid.append({**digest(path), "error": str(exc)})
        continue
    config = d["config"]
    train, steering = config["train"], config.get("steering", {})
    command = shlex.split(d["cmd"]) if isinstance(d["cmd"], str) else d["cmd"]
    overrides = dict(x.lstrip("+").split("=", 1) for x in command if "=" in x)
    estimator = overrides.get("algorithm.adv_estimator")
    manifests.append({
        **digest(path), "run": path.parent.name, "timestamp": d["ts"],
        "config_estimator": train.get("adv_estimator"), "command_estimator": estimator,
        "estimator_matches": estimator == train.get("adv_estimator"),
        "std_normalization": overrides.get("algorithm.norm_adv_by_std_in_grpo"),
        "kl_in_reward": overrides.get("algorithm.use_kl_in_reward"),
        "reference_kl_loss": overrides.get("actor_rollout_ref.actor.use_kl_loss"),
        "loss_aggregation": overrides.get("actor_rollout_ref.actor.loss_agg_mode"),
        "rollout_n": train.get("rollout_n"), "steering": steering.get("enabled"),
        "steered_fraction": steering.get("steered_fraction"),
        "w_beh_mode": steering.get("w_beh_mode"),
        "alpha_initial": steering.get("steered_adv_weight"),
        "alpha_decay_steps": steering.get("steered_adv_decay_steps"),
        "reward_from_score_recorded": train.get("reward_from_score", "ABSENT"),
        "opd_command_options": [x for x in command if "distillation." in x or "mix_opd" in x],
        "code_revision_recorded": d.get("git_commit", "ABSENT"),
        "evidence_level": "launch_configuration_only",
    })

logs, metrics = [], []
selected = ["actor/pg_loss", "actor/loss", "actor/grad_norm", "actor/pgs_steered_alpha",
            "critic/score/mean", "critic/rewards/mean", "rollout_extra/won/mean",
            "rollout_extra/score/mean", "timing_s/update_actor"]
for root in (ROOT / "outputs", ROOT / "runs", ROOT / "logs"):
    for path in sorted(root.rglob("*.log")):
        steps, distillation_lines = [], []
        for line_no, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            match = re.search(r"\bstep:(\d+)\s+-", line)
            if "distillation/loss" in line:
                distillation_lines.append(line_no)
            if not match:
                continue
            step = int(match.group(1))
            steps.append(step)
            values = {}
            for key in selected:
                found = re.search(re.escape(key) + r":([-+0-9.eE]+)(?:\s|$)", line)
                if found:
                    values[key] = float(found.group(1))
            metrics.append({"path": str(path), "line": line_no, "step": step, **values})
        logs.append({**digest(path), "step_count": len(steps), "steps": steps,
                     "distillation_metric_lines": distillation_lines})

ssb_rows = [r for r in metrics if r["path"].endswith("classic_ssb3b_pgs/launch.log")]
steps = [r for r in ssb_rows if all(k in r for k in [
    "actor/loss", "actor/pg_loss", "actor/pgs_steered_alpha", "actor/grad_norm"])]
pg_loss_errors = [abs(r["actor/loss"] - r["actor/pg_loss"]) for r in steps]
ssb_config = [r for r in manifests if r["run"] == "classic_ssb3b_pgs"][-1]
decay_steps, alpha0 = ssb_config["alpha_decay_steps"], ssb_config["alpha_initial"]
alpha_errors = [abs(r["actor/pgs_steered_alpha"] - alpha0 * 0.5 *
                   (1 + math.cos(math.pi * min(r["step"], decay_steps) / decay_steps))) for r in steps]
checks = {
    "all_legacy_blobs_match": all(r["blob_matches"] for r in legacy),
    "all_current_snapshots_match": all(r["matches_current"] for r in source_records),
    "all_manifest_estimators_match_commands": all(r["estimator_matches"] for r in manifests),
    "no_local_launch_has_opd_options": all(not r["opd_command_options"] for r in manifests),
    "ssb_observed_steps": len(steps),
    "ssb_metric_rows_without_complete_actor_metrics": [r["step"] for r in ssb_rows if r not in steps],
    "ssb_actor_loss_equals_pg_loss_max_absolute_error": max(pg_loss_errors, default=None),
    "ssb_alpha_decay_steps_from_manifest": decay_steps,
    "ssb_alpha_matches_manifest_schedule_max_absolute_error": max(alpha_errors, default=None),
    "ssb_all_observed_grad_norms_positive": bool(steps) and all(r.get("actor/grad_norm", 0) > 0 for r in steps),
    "main_ppo_logs": sum(Path(r["path"]).name == "main_ppo.log" for r in logs),
    "nonempty_main_ppo_logs": sum(Path(r["path"]).name == "main_ppo.log" and r["bytes"] > 0 for r in logs),
    "valid_manifests": len(manifests), "invalid_manifests": len(invalid),
    "runs_with_valid_manifests": len({r["run"] for r in manifests}),
}
write("objective-inventory.json", {"date": "2026-10-23", "manifests": manifests,
    "invalid_manifests": invalid, "logs": logs, "current_sources": source_records,
    "legacy_sources": legacy, "checks": checks,
    "scope": "Implementation/configuration audit with partial runtime corroboration; no full historical run reconstruction."})
write("runtime-metrics.json", metrics)
with (HERE / "run-matrix.csv").open("w", newline="") as handle:
    fields = ["run", "timestamp", "command_estimator", "std_normalization", "steering",
              "steered_fraction", "w_beh_mode", "alpha_decay_steps", "loss_aggregation",
              "reward_from_score_recorded", "path"]
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(manifests)
print(json.dumps(checks, ensure_ascii=False, indent=2))
assert checks["all_legacy_blobs_match"], "Retrieved historical bytes differ from Git blobs"
assert checks["all_manifest_estimators_match_commands"]
# Live source drift is reported, not folded into the frozen historical audit.
assert steps and max(pg_loss_errors) < 1e-7 and max(alpha_errors) < 1e-12
