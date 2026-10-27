A/B/C/D frozen-checkpoint diagnostic

Status: independent cloud job queued after CPU-only preflight. GPU sampling has
NOT started; no new success rates or advantage-group results exist yet.
Cloud root: /root/trace-webshop-arm-ratio/experiments/skill_abcd_20261023
Queue PID: 1640270

A: no Skill, 8 bare samples per group.
B: original archived researcher Skill, 4 SPS + 4 bare samples per group.
C: corrected procedure plus prior-episode candidate/option hints, 4 SPS + 4 bare.
D: same corrected procedure with that hint paragraph removed, 4 SPS + 4 bare.

Frozen original Qwen2.5-3B-Instruct; no weight updates. Binary reward int(won).
Advantages = reward - same-task group mean, without std normalization.
Nonzero-advantage group iff 0 < wins < 8. Environment errors and incomplete
groups are explicitly incomplete, never silently counted as model failures.

Tasks 00510 and 00512; four groups/task/arm, eight samples/group = 256 episodes.
Eight additional bare episodes calibrate one pooled entropy threshold, frozen
for B/C/D. Same SPS kernel from the original frozen source, same sampling
T=1, top_p=1, top_k=-1. At most 15 turns, 1024 generated tokens/turn, 16384
response-stream budget. Independent sequential SGLang engine after GPU release.
Same task/group/sample seeds across arms; bare rows are paired correlated
controls, not independent observations for ordinary binomial inference.

Outputs: summary.json (overall/per-task success and nonzero-advantage groups,
guided/bare breakdown); group_advantages.json (eight rewards and advantages
per group); episodes.jsonl (observable actions and observations, no private
reasoning export); calibration_summary.json; checkpoint_verified.json.

Isolation: copied source/configuration, new Skills and output directory.
No connection to the live Ray cluster or new researcher requests. Original
28 frozen file hashes were verified unchanged. No original job stopped,
reconfigured or restarted. GPU 1 must be empty with >=35000 MiB free for
three checks AND original sequence must be completed_analysis before launch.
Wait timeout: 48 hours. Sampling timeout: 12 hours. Only the owned process
group is terminated on sampling timeout. Actual GPU sampling is not yet tested.

Scope: posthoc diagnostic on two inspected training cases, not representative
evaluation, not evidence of training/generalization improvement. B-C combines
correction, concision, wording and changed hints. C-D also changes prompt
length; no length-matched placebo. No claim that privilege causes poor results.
