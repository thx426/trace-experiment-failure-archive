# PGS compact3072 正式实验

用户批准：研究员每次输入不超过 3072 token、服务并发 16、先 All-Skill 后 Selective-Skill，不做小试。

## 已部署的实现

- 新目录：`/root/trace-webshop-arm-ratio/experiments/pass5_compact3072_20261024_r2`。
- 每轮构建短状态，保留完整任务、预算、真实实验结果；证据按 token 摘取。完整请求和实验轨迹继续归档。
- 发送前调用研究员服务原生 `/tokenize`，计入聊天模板和工具定义，超过 3072 则拒发；返回 usage 再核对。
- 目标输入不超过 2500 token，硬限制 3072；fetch_raw 每页包含头部不超过 512 token。
- 实验预算耗尽后从下一轮工具定义中撤下 run_experiment；文档目标 300–400 词、硬上限 500 词，最多一次修订。
- 研究员服务 max_num_seqs=16，max_model_len=7168，输出预算保留 4096。
- 两组均从同一个 Qwen2.5-3B-Instruct 初始 checkpoint 开始，各训练 100 步，seed=42，奖励、训练与评测数据、优化器、采样和引导比例保持原配置。
- All-Skill 完成并通过完整性审计后自动启动 Selective-Skill；失败、超限、OOM 或持续抢占会停止，不静默回退配置。

## 已完成校验

本地与云端 6 项 CPU 单元测试通过。云端真实研究员 tokenizer 的 Unicode/长文本截断校验通过，两组与原配置的学生训练、奖励、采样和评测参数一致性检查通过。没有试跑生成、环境轨迹或更新参数。

## 状态

2026-10-24 00:46（Asia/Shanghai）的首次启动在学生初始化时失败：SGLang preload 选择了系统 CUDA 运行库，缺少 cudaGetDriverEntryPointByVersion。该次无 rollout、无优化器更新，两组 GPU owner 均验证释放。失败日志保留在无 `_r2` 后缀的原目录。

已为学生子进程补齐原项目 activate_experiment.sh 的环境设置，使训练 venv 的 CUDA runtime 优先加载。研究代码、checkpoint、奖励和实验配置未改变。修复后的 `_r2` 正式序列已启动，PID 1969791；All-Skill owner PID 1970281，研究员 owner PID 1969840。当前 All-Skill 已进入训练前无 Skill 评测（保留原 val_before_train=true）。研究员原生模板计数 736 token，服务日志确认并发上限 16。

云端状态：`sequence_status.json`；序列日志：`sequence.log`；研究员日志：`/tmp/p5compact3072-research24r2/command.log`；All-Skill 日志：`/tmp/p5compact3072-all24r2/command.log`。

这属于新的 PGS 输入策略版本。可用于与原 PGS 比较，但不能把原先中断的 All-Skill 步数和新版拼接成一条训练轨迹。
