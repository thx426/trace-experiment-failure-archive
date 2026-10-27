"""Read-only analysis of archived step-20 evaluations. No model/service calls."""
import json
import math
import re
import random
import statistics
from collections import Counter
from html import escape
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
E = json.loads((ROOT / 'evidence.json').read_text())
T = json.loads((ROOT / 'related_training_skills.json').read_text())
INDEX = {x['task_id']: x for x in E['index']}
BINS = ['zero', 'low_positive', 'middle', 'high']
LABEL = {'zero': '初始 pass@5 = 0', 'low_positive': '0 < 初始 pass@5 < 0.5',
         'middle': '0.5 ≤ 初始 pass@5 < 0.8', 'high': '初始 pass@5 ≥ 0.8'}

# Representative episodes were chosen for explanatory clarity AFTER random task
# sampling. All five episodes per arm are retained in the report to avoid hiding
# within-task variation. Session IDs are zero-based; action steps are one-based.
NOTES = [
 dict(id='04021', grpo_session=1, all_session=2, category='规格漏选 / 提前购买',
      diagnosis='GRPO 选择 39x49in 和 thanksgivinggoo9948 后购买；All-Skill 在同一商品页第 3 步直接购买，奖励只有 0.6。',
      other='All-Skill 的 session 1、2、3、4 都漏选颜色和尺寸后购买；session 0 使用了错误商品 ID 并重复无效购买。',
      correction='第 3 步先 click[39x49in]，再 click[thanksgivinggoo9948]，最后 click[buy now]。搜索词含有规格不等于规格已选中。'),
 dict(id='00805', grpo_session=2, all_session=3, category='已配置商品但不购买',
      diagnosis='两条轨迹前 5 步动作与返回观测完全相同；GRPO 第 6 步购买成功，All-Skill 第 6 步继续选色，颜色/尺寸交替重复至第 15 步。',
      other='另有翻页循环、切换错误尺寸、拼接颜色和尺寸为无效点击标签等失败。不能把全部 5 次失败都归为同一条循环。',
      correction='第 6 步改为 click[buy now]。已执行过的有效选项动作应记入状态；页面文字未变化时，不要反复选择同一规格。'),
 dict(id='04965', grpo_session=0, all_session=0, category='规格漏选 / 放弃已配置商品',
      diagnosis='两组到达相同商品；GRPO 选 20x24in+20x48in 和 rabbit3555lgoo1957 后成功，All-Skill 第 3 步直接买，奖励 0.6。',
      other='All-Skill session 1 已选对尺寸和颜色，却再次选尺寸并返回搜索；其余尝试存在搜索、翻页循环。',
      correction='session 0 第 3 步补齐尺寸和颜色；session 1 选齐后立即购买，不要退回搜索。'),
 dict(id='02762', grpo_session=2, all_session=0, category='漏选一个颜色属性',
      diagnosis='All-Skill 选择 youth 和 xx-large，却从未选择 cranberry，第 6 步购买只得 10/11 = 0.9091。',
      other='另外 4 次均未完成购买，存在搜索循环和无效操作。',
      correction='第 6 步购买前插入 click[cranberry]。不要把高部分奖励当作完全满足约束。'),
 dict(id='04603', grpo_session=3, all_session=1, category='选齐后离开商品页',
      diagnosis='All-Skill 已找到正确商品并选好 red、large，第 5 步却点 < prev，随后循环翻页至 15 步。GRPO 最终购买成功。',
      other='GRPO 成功轨迹也不是完美脚本：第 5 步点击了一个当前不可用的商品 ID，无效且未改变页面，随后第 6 步购买。',
      correction='All-Skill 第 5 步应 click[buy now]。有完整规格记录时，停止继续浏览。'),
 dict(id='02272', grpo_session=0, all_session=0, category='漏选单个颜色属性',
      diagnosis='搜索和打开商品相同；GRPO 第 3 步选 butterfly，All-Skill 第 3 步直接买，奖励 0.75。',
      other='All-Skill session 3 同样直接买；其他尝试在搜索或翻页中耗尽回合。',
      correction='第 3 步改为 click[butterfly]，再购买；保留原始可点击标签。'),
 dict(id='04495', grpo_session=1, all_session=2, category='漏选精确颜色编码',
      diagnosis='All-Skill 到达正确鞋款并选 11 码，却漏选 r.brown-hole1887；第 4 步购买奖励 6/7 = 0.8571。',
      other='其他尝试还出现把精确颜色泛化为 brown 后选择其他鞋款、以及退回结果页后无效购买。',
      correction='第 4 步先 click[r.brown-hole1887]，再购买。编码是任务约束，不能只保留 brown。'),
 dict(id='06578', grpo_session=4, all_session=1, category='漏选颜色和尺寸',
      diagnosis='All-Skill 连选两次 men 后第 5 步购买，漏掉 kelly green 和 2t，奖励 0.8；GRPO 最终选齐三类规格并成功。',
      other='其余 4 次没有完成购买。GRPO 成功轨迹同样有一次重复 men，因此重复动作本身不必然失败，关键是是否完成剩余规格并购买。',
      correction='第 4 步不再重复 men，依次 click[2t]、click[kelly green]、click[buy now]。'),
 dict(id='05944', grpo_session=1, all_session=0, category='重复选择 / 未完成购买',
      diagnosis='All-Skill 已到达正确商品，第 3—15 步连续点击 xnj-tshirt333-black 共 13 次，既没选 x-large，也没购买。',
      other='另外几次还存在丢弃精确编码后的搜索、翻页和返回搜索循环。',
      correction='第 4 步应 click[x-large]，第 5 步 click[buy now]；颜色有效点击一次后转入下一项。'),
 dict(id='06036', grpo_session=4, all_session=2, category='提前购买 / 选齐后转错商品',
      diagnosis='所示 All-Skill 轨迹在第 4 步已选齐 medium、red，第 5 步却返回搜索，换到男士短裤，最后奖励 0。',
      other='session 0、1、4 仅选 medium 就购买，均为 6/7；session 3 连颜色和尺寸都没选，奖励 5/7。这一高初始成功率层只有 1 道候选题。',
      correction='session 2 第 5 步直接购买；另外三条只选尺寸的轨迹在购买前补 click[red]。'),
]
for x in NOTES:
    x['task_id'] = 'webshop_' + x['id']
    x['index'] = INDEX[x['task_id']]

def episode(tid, arm, sid):
    return next(r for r in E['candidate_episodes'][tid][arm] if r['session_id'] == sid)

def visits(row):
    found = []
    for s in row['trajectory']['steps']:
        m = re.fullmatch(r'click\[([a-z0-9]{10})\]', s['action_norm'])
        if m and s['valid'] and s['acted'] and 'buy now' in s['observation'].lower():
            found.append(m[1])
    return found

def bought(row):
    return any(s['valid'] and s['acted'] and s['action_norm'] == 'click[buy now]'
               for s in row['trajectory']['steps'])

def aggregate(ids, arm):
    rows = [r for tid in ids for r in E['candidate_episodes'][tid][arm]]
    return dict(attempts=len(rows), wins=sum(r['won'] for r in rows),
                valid_purchases=sum(bought(r) for r in rows),
                no_valid_purchase=sum(not bought(r) for r in rows),
                any_invalid_action=sum(r['quality']['invalid_action_count'] > 0 for r in rows),
                any_repeated_action_observation=sum(r['quality']['repeated_action_observation_count'] > 0 for r in rows),
                mean_env_steps=statistics.mean(r['env_steps'] for r in rows),
                mean_score=statistics.mean(r['score'] for r in rows),
                environment_errors=sum(r['outcome'] == 'ENV_ERROR' for r in rows),
                token_truncated=sum(r['truncated'] for r in rows))

def target_visits(ids):
    result = dict(tasks=len(ids), attempts=len(ids)*5, tasks_reached_reference_product=0,
                  attempts_reached_reference_product=0, purchases_on_reference_product=0,
                  purchases_on_other_product=0)
    for tid in ids:
        ref = {visits(r)[-1] for r in E['candidate_episodes'][tid]['grpo'] if r['won'] and visits(r)}
        all_rows = E['candidate_episodes'][tid]['all_skill']
        flags = [bool(ref.intersection(visits(r))) for r in all_rows]
        result['tasks_reached_reference_product'] += any(flags)
        result['attempts_reached_reference_product'] += sum(flags)
        for r in all_rows:
            if bought(r):
                on_ref = bool(visits(r)) and visits(r)[-1] in ref
                result['purchases_on_reference_product' if on_ref else 'purchases_on_other_product'] += 1
    return result

candidate = E['candidate_ids']
selected = E['selected_ids']
assert [x['task_id'] for x in NOTES] == selected
assert len(candidate) == 46 and len(selected) == 10 and not E['train_dev_overlap']
rng = random.Random(E['sampling_seed'])
sample_check = []
for b in BINS:
    pool = sorted(tid for tid in candidate if INDEX[tid]['sampling_bin'] == b)
    sample_check.extend(rng.sample(pool, min(3, len(pool))))
assert selected == sample_check
for x in E['index']:
    p = 1 - math.comb(x['base_n']-x['base_c'], 5)/math.comb(x['base_n'], 5) if x['base_n']-x['base_c'] >= 5 else 1
    assert abs(p-x['base_pass5']) < 1e-12
for tid in candidate:
    assert INDEX[tid]['grpo_successes'] > 0 and INDEX[tid]['all_skill_successes'] == 0
    for arm in ['grpo', 'all_skill']:
        rows = E['candidate_episodes'][tid][arm]
        assert len(rows) == 5 and {r['session_id'] for r in rows} == set(range(5))
        assert all(r['global_step'] == 20 and r['evaluation_bare'] for r in rows)

overall = {}
for arm, tasks in E['evaluations'].items():
    successes = sum(sum(v) for v in tasks.values())
    solved = sum(any(v) for v in tasks.values())
    overall[arm] = dict(tasks=len(tasks), attempts=sum(len(v) for v in tasks.values()),
                        successes=successes, solved=solved, pass1=successes/1280, pass5=solved/256)
strata = []
for b in BINS:
    rows = [x for x in E['index'] if x['sampling_bin'] == b]
    outcomes = Counter(x['outcome'] for x in rows)
    strata.append(dict(bin=b, label=LABEL[b], population=len(rows), **outcomes,
                       selected=[tid for tid in selected if INDEX[tid]['sampling_bin'] == b]))
g8 = episode('webshop_00805', 'grpo', 2)['trajectory']['steps']
a8 = episode('webshop_00805', 'all_skill', 3)['trajectory']['steps']
identical_prefix = all((g['action_norm'], g['observation']) == (a['action_norm'], a['observation'])
                       for g, a in zip(g8[:5], a8[:5]))
assert identical_prefix
assert len({s['observation'] for s in a8[1:5]}) == 1

skill_notes = [
 (16, '01263', '错误地将检索当作全属性过滤，并建议购买首个商品',
  '文档要求在结果页立即对第一个商品 Buy Now，并假设它符合条件。结果页需要先打开商品，颜色/尺寸仍需实际点击，不能由查询词代替。该训练组两个实际引导轨迹只选 medium 不选 white，奖励均为 6/7；另一引导轨迹和一条 bare 轨迹选齐后成功。此训练题与评测 06036 同商品、不同颜色。'),
 (16, '04959', '将合法精确编码判断成疑似错误；缺少规格点击的完整收尾',
  '文档把 xnj-tshirt336-black 称为 likely erroneous，并建议初次搜索忽略它。宽化初次检索本身可以合理，但不能据此删除最终颜色约束。同组 bare session 7 成功选择该编码；引导 session 1 选齐后循环，session 2 漏尺寸只得 8/9。与评测 05944 同商品、不同颜色。'),
 (12, '01764', '文字提醒选规格，仍没有形成可靠动作与完全成功样本',
  '文档已经强调选好颜色和尺寸再购买，但这组 8 条学生轨迹完全成功数仍为 0；奖励是 0.8、六个 0.6、0。四条分配给引导的轨迹都有实际 steering，仍然漏选颜色。文档把精确色号简化成 Wood Grain，还引用了具体品牌；不能把“有提醒”当作学生已获得可执行、可迁移能力。'),
 (4, '00857', '失败叙述替代了可执行指导',
  '文档只有任务属性和六次失败的概括，没有明确下一步或有效恢复路径；四条引导轨迹均产生实际 steering，但奖励全为 0。将这种文字作为 Skill，并不保证得到正面的探索信号。'),
 (12, '02019', '同类任务上的相反检索建议（对照证据）',
  '这份毛衣文档反而建议优先精确编码 xnj-tshirt349-wine，并报告宽泛检索失败。不同任务的有效检索策略可以不同，不能仅凭这两份文档认定逻辑矛盾；它们说明应把检索建议写成有观测条件的分支，而非通用绝对规则。文档所称研究员成功尚未在本报告重新核验。'),
]
skills = []
for step, short, title, comment in skill_notes:
    group = next(x for x in T['related_training_sessions'] if x['training_step'] == step and x['task_id'].endswith(short))
    mean = statistics.mean(r['score'] for r in group['rollouts'])
    skills.append(dict(training_step=step, task_id=group['task_id'], title=title, comment=comment,
                       group=group, raw_score_group_mean=mean,
                       raw_score_minus_mean=[dict(session_id=r['session_id'], value=r['score']-mean) for r in group['rollouts']]))

derived = dict(checkpoint_step=20, sampling_seed=E['sampling_seed'], overall=overall,
               paired_outcomes=dict(Counter(x['outcome'] for x in E['index'])), strata=strata,
               candidates=[INDEX[x] for x in candidate], cases=NOTES,
               candidate_metrics={a:aggregate(candidate,a) for a in ['grpo','all_skill']},
               sample_metrics={a:aggregate(selected,a) for a in ['grpo','all_skill']},
               candidate_target_visits=target_visits(candidate), sample_target_visits=target_visits(selected),
               training_summary=E['training_summary'], skill_analysis=skills,
               exact_prefix_00805={'grpo_session':2,'all_skill_session':3,'identical_actions_and_observations_steps':5,
                                  'verified':identical_prefix},
               limitations=['Single seed; five evaluation attempts per task; paired outcome is not deterministic capability.',
                            'Candidate cohort is conditioned on GRPO success and All-Skill failure; aggregate markers are descriptive, not global treatment effects.',
                            'Sampled evaluation tasks are dev tasks, disjoint from training. Related training docs are linked by products, not the same task.',
                            'Raw-score dispersion and score-minus-mean are diagnostic proxies; no actual gradient or optimizer-inclusion audit is claimed.',
                            'Skill statements about researcher experiments are unverified claims unless explicitly cross-checked against student actions.',
                            'No new training or evaluation was run. Experiment and monitor remain paused.'])
(ROOT/'analysis.json').write_text(json.dumps(derived, ensure_ascii=False, indent=2))

def esc(v): return escape(str(v), quote=True)
def pct(v): return f'{v*100:.2f}%'
def table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>' + ''.join('<th>'+esc(x)+'</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows) + '</tbody></table></div>'
def cell(v): return esc(v)
def actions(row):
    return '<ol class="actions">'+''.join('<li><code>'+esc(s['action_norm'])+'</code>'+(' <span class="bad">无效</span>' if not s['valid'] else '')+'</li>' for s in row['trajectory']['steps'])+'</ol>'
def episode_detail(row):
    head=f"session {row['session_id']} · reward {row['score']:.4f} · {'成功' if row['won'] else '失败'} · {row['env_steps']} 个环境步"
    observation='<details><summary>逐步环境观测与动作有效性</summary><pre>'+esc(row['trajectory']['initial_obs'])+'</pre>'
    for s in row['trajectory']['steps']:
        observation+=f'<details><summary>第 {s["t"]} 步 · {esc(s["action_norm"])} · valid={s["valid"]}</summary><pre>{esc(s["observation"])}</pre></details>'
    observation+='</details>'
    source='<p class="source">原始记录：'+esc(row['source_path'])+'<br>SHA-256：'+esc(row['source_sha256'])+'</p>'
    return '<details class="episode"><summary>'+esc(head)+'</summary>'+actions(row)+observation+source+'</details>'

parts = ['''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GRPO 胜出题目的分层分析 · step 20</title><style>
:root{color-scheme:light;--ink:#182c3b;--muted:#5a6d7b;--line:#d8e1e6;--bg:#f4f7f8;--accent:#096b70}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.7 system-ui,-apple-system,"PingFang SC",sans-serif}main{max-width:1180px;margin:auto;padding:40px 28px 80px}h1{font-size:34px;line-height:1.3;margin:8px 0 16px}h2{margin-top:42px;font-size:25px}h3{font-size:21px;margin-top:0}p{margin:10px 0}a{color:var(--accent)}.eyebrow{color:var(--accent);font-size:13px;font-weight:700;letter-spacing:1px}.lead{font-size:20px;max-width:1000px}.note,.source{color:var(--muted);font-size:14px}.callout{border-left:4px solid var(--accent);background:#e5f1f1;padding:16px 20px;margin:24px 0}.case,.skill{background:white;border:1px solid var(--line);border-radius:10px;padding:24px;margin:22px 0}.tag{color:var(--accent);font-weight:650}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}.pair>div{min-width:0;background:#f6f8fa;border-radius:8px;padding:16px}table{border-collapse:collapse;width:100%;font-size:14px;background:white}th,td{border-bottom:1px solid var(--line);padding:10px 12px;text-align:left;vertical-align:top}th{background:#e9eff2;white-space:nowrap}.table-wrap{overflow:auto;margin:18px 0}code{font-size:13px;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f8;border-radius:6px;padding:14px;font-size:13px;line-height:1.6}.actions{padding-left:28px}.actions li{padding:2px 0}summary{cursor:pointer;font-weight:600}details{margin:10px 0}.episode{padding:10px 12px;border:1px solid var(--line);border-radius:6px}.bad{color:#a43726;font-size:12px}.source{overflow-wrap:anywhere}.jump{display:flex;gap:12px;flex-wrap:wrap}.pill{border:1px solid var(--line);border-radius:30px;padding:4px 12px;text-decoration:none}.foot{border-top:1px solid var(--line);padding-top:18px;margin-top:36px}@media(max-width:720px){main{padding:24px 14px}.pair{grid-template-columns:1fr}.case,.skill{padding:16px}h1{font-size:28px}}@media print{body{background:white}.case{break-inside:avoid}.pair{display:block}details{break-inside:avoid}}</style></head><body><main>
<div class="eyebrow">WebShop · 2026-10-24 · 既有日志分析 · 实验保持暂停</div>
<h1>GRPO 答对、All-Skill 未答对：<br>按初始 pass@5 分层抽样分析</h1>
<p class="lead">主要分叉出现在找到商品之后：漏选规格就购买，或已经选齐却重复操作、返回搜索。现有证据不足以把性能差异统一归因于“特权信息”。</p>
<div class="callout">比较双方 <b>step 20</b> 的完整 <b>无 Skill</b> 评测：相同 256 道 dev 题，每题 5 次，温度 0.6。All-Skill 后来完成 35 次更新后被用户暂停，但最新完整评测及已保存 checkpoint 为 20，所以不把 GRPO step 100 与它混作同进度对照。本次没有启动模型、训练或额外评测。</div>
<p class="note">“All-Skill”表示所有训练任务均有研究员资格；每组 8 条轨迹中 4 条分配给引导、4 条 bare。评测时两组均无 Skill。以下呈现外显动作与环境观测，不重构模型内部思维链。</p>''']
parts.append('<h2>比较口径与总体结果</h2>')
parts.append(table(['模型 / checkpoint','单次成功率估计（5 次均值）','pass@5（至少一次成功）'],[
 ['传统 GRPO / 20',f"265/1280 = {pct(overall['grpo']['pass1'])}",f"130/256 = {pct(overall['grpo']['pass5'])}"],
 ['All-Skill / 20',f"214/1280 = {pct(overall['all_skill']['pass1'])}",f"119/256 = {pct(overall['all_skill']['pass5'])}"]]))
parts.append('<p>配对结果：双方都成功 84 题；仅 GRPO 成功 <b>46 题</b>；仅 All-Skill 成功 <b>35 题</b>；双方都失败 91 题。净差是 11 题，而不是把 46 题全部解释为确定退化。46 题中有 23 题 GRPO 也只成功 1/5；采样波动不可忽略。</p>')
parts.append('<p>筛选条件是当前 GRPO 至少一次成功、All-Skill 五次全失败。因此按“当前 GRPO pass@5”无法再区分它们（全部为 1）。分层改用训练前 bare 模型每题 20 次尝试的估计：<code>pass@5 = 1 − C(20−c,5)/C(20,5)</code>。低层进一步拆为零成功与低正值，固定随机种子 20261024，每层不放回抽取 min(3, N) 题。</p>')
parts.append(table(['初始层','全体题数','GRPO 独有成功','All-Skill 独有成功','抽取题目'],[
 [esc(s['label']),s['population'],s.get('grpo_only',0),s.get('all_only',0),', '.join('<a href="#case-'+t[-5:]+'">'+t[-5:]+'</a>' for t in s['selected'])] for s in strata]))
parts.append('<p class="note">原始 low 层占总体 221/256 = 86.3%，占候选 42/46 = 91.3%。难题数量多并不自动证明该层更容易被 Skill 损害。这里是诊断抽样，不能由 10 题直接估计总体因果效应。</p>')
parts.append('<h2>46 题全量轨迹扫描</h2>')
cm=derived['candidate_metrics']
parts.append(table(['指标（仅这 46 题）','GRPO，230 次','All-Skill，230 次'],[
 ['未执行有效购买',cm['grpo']['no_valid_purchase'],cm['all_skill']['no_valid_purchase']],
 ['出现至少一个无效动作',cm['grpo']['any_invalid_action'],cm['all_skill']['any_invalid_action']],
 ['出现重复的动作—观测对',cm['grpo']['any_repeated_action_observation'],cm['all_skill']['any_repeated_action_observation']],
 ['平均环境步数',f"{cm['grpo']['mean_env_steps']:.2f}",f"{cm['all_skill']['mean_env_steps']:.2f}"]]))
tv=derived['candidate_target_visits']
parts.append(f'<p>All-Skill 在 <b>{tv["tasks_reached_reference_product"]}/46 题</b>中至少有一次有效打开了 GRPO 成功购买的商品，共 {tv["attempts_reached_reference_product"]}/230 次尝试。82 次有效购买中 57 次发生在这些参考商品上。因此，至少这部分失败不能简单解释为检索不到商品。</p>')
parts.append('<p class="note">这些指标是在“GRPO 成功、All-Skill 失败”的条件下计算，存在筛选偏差，不能当作全体任务上的处理效应。参考商品由 GRPO 成功轨迹的最后一次有效商品点击识别；“到达参考商品”不代表已选齐规格。日志中本候选集没有 ENV_ERROR 或 token 截断；耗尽 15 回合不等于 HTTP 超时。</p>')
parts.append('<h2>10 道分层样本：失败分叉与人工纠正点</h2><p>每题保留双方全部 5 次轨迹；主面板选取最便于定位分叉的一条成功/失败作解释，并非声称其他尝试行为相同。session 从 0 编号，动作从第 1 步编号。人工纠正是依据已成功轨迹提出的操作建议，本次没有新执行回放验证。</p>')
parts.append(table(['题号','初始 pass@5','GRPO 成功数','主要失败','纠正点'],[
 ['<a href="#case-'+x['id']+'">'+x['id']+'</a>',pct(x['index']['base_pass5']),str(x['index']['grpo_successes'])+'/5',esc(x['category']),esc(x['correction'])] for x in NOTES]))
for x in NOTES:
    tid=x['task_id']; g=episode(tid,'grpo',x['grpo_session']); a=episode(tid,'all_skill',x['all_session'])
    parts.append(f'<section class="case" id="case-{x["id"]}"><div class="tag">{esc(LABEL[x["index"]["sampling_bin"]])} · 初始 {x["index"]["base_c"]}/20 成功 · pass@5 {pct(x["index"]["base_pass5"])}</div><h3>{x["id"]} · {esc(x["category"])}</h3>')
    parts.append('<details><summary>题目原文</summary><pre>'+esc(g['trajectory']['initial_obs'])+'</pre></details>')
    parts.append('<p><b>观察：</b>'+esc(x['diagnosis'])+'</p><p><b>其他尝试：</b>'+esc(x['other'])+'</p><p><b>人工纠正：</b>'+esc(x['correction'])+'</p>')
    parts.append(f'<div class="pair"><div><b>GRPO · session {x["grpo_session"]} · reward {g["score"]:.4f}</b>'+actions(g)+f'</div><div><b>All-Skill · session {x["all_session"]} · reward {a["score"]:.4f}</b>'+actions(a)+'</div></div>')
    if x['id']=='00805':
        parts.append('<div class="callout">已逐字核验：两条轨迹前 5 步动作及观测完全相同。打开商品及选择颜色、尺寸后，返回的页面文字也相同；因此必须利用已执行的动作记录保存选择状态。这是共享环境的可观测性限制，不能单独解释为 All-Skill 的环境故障。</div>')
    parts.append('<details><summary>展开全部 10 条轨迹及原始来源</summary>')
    for arm,title in [('grpo','GRPO'),('all_skill','All-Skill')]:
        parts.append('<h4>'+title+'</h4>')
        parts.extend(episode_detail(r) for r in sorted(E['candidate_episodes'][tid][arm],key=lambda r:r['session_id']))
    parts.append('</details></section>')

parts.append('<h2>回看训练 Skill：有哪些实际证据？</h2><p>评测题与训练题 ID 不重合，所以评测时不存在“这道题收到的那份 Skill”。以下只选训练中访问相同商品、且题意相近的档案作机制诊断。产品相同不代表任务相同，也不能证明训练文档对评测错误的因果作用。</p>')
parts.append('<p>前 20 步共 320 个任务—训练步组：<b>170 组（53.1%）没有文档，150 组有非空文档</b>。分配给引导的 1280 条轨迹中包括缺文档后回退为 bare 的情况，不能把臂分配等同于实际注入。相关训练档案里，有文档的 192 条引导臂轨迹中 176 条记录了正的 steer_count。<code>guidance_chars=0</code> 也不能单独证明没引导，因为这里的 Skill 通过 steering 分支影响生成；需看实际 steering 账本。</p>')
for s in skills:
    group=s['group']; rows=sorted(group['rollouts'], key=lambda r:r['session_id'])
    parts.append(f'<section class="skill"><div class="tag">训练 step {s["training_step"]} · {s["task_id"]}</div><h3>{esc(s["title"])}</h3><p>{esc(s["comment"])}</p>')
    parts.append(table(['session','分配臂','实际 steering 位置数','奖励','完全成功'],[
      [r['session_id'],'引导' if r['steering']['steer_arm'] else 'bare',r['steering']['steer_count'],f"{r['score']:.4f}",'是' if r['won'] else '否'] for r in rows]))
    parts.append('<details><summary>原始 Skill 全文、任务及来源</summary><pre>'+esc(group['initial_obs'])+'</pre><pre>'+esc(group['doc'])+'</pre><p class="source">'+esc(group['doc_path'])+'<br>SHA-256：'+esc(group['doc_sha256'])+'</p></details>')
    parts.append('<details><summary>本训练组 8 条学生动作</summary>')
    for r in rows:
        parts.append(f'<details><summary>session {r["session_id"]} · reward {r["score"]:.4f}</summary><ol class="actions">'+''.join('<li><code>'+esc(a['action'])+'</code>'+(' <span class="bad">无效</span>' if not a['valid'] else '')+'</li>' for a in r['actions'])+'</ol><p class="source">'+esc(r['source_path'])+'</p></details>')
    parts.append('</details></section>')

parts.append('''<h2>与 GRPO 信号的联系：不要把“未成功”误读为“没优势”</h2>
<p>训练 step 12 / 01764 全部未完全成功，但原始奖励为 <b>[0.8, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0]</b>，均值 0.55。漏选颜色的 0.8 分轨迹高于均值 0.25；直接购买的 0.6 分轨迹也高于均值 0.05。在当前共享组均值的 GRPO 配置下，这说明相对奖励可偏好“能买到部分匹配商品”的轨迹，未必要求它完成全部规格。</p>
<p>这里的差值只是从原始 score 重算的诊断量，<b>不冒充实际保存的 advantage、token 权重、mask 或最终梯度</b>。完整参数更新归因还需对齐奖励张量、训练审计和优化器记录。本报告未作该层级审计，也未改变奖励。</p>
<p>前 20 步具有组内原始奖励差异的比例：GRPO <b>302/320 = 94.4%</b>，All-Skill <b>287/320 = 89.7%</b>；这是优势产生条件的代理统计，不等于实测有效梯度比例。All-Skill 指定 bare 臂完全成功 93/1280，引导臂 99/1280；传统 GRPO 为 215/2560。引导目前并未提供很大的完整成功增量。</p>
<h2>现在可以下的判断与不能下的判断</h2>
<p><b>已支持：</b>这批样本的常见行为失败是规格约束没有转化成实际点击，以及完成规格后没有及时购买。训练 Skill 存在可核验的错误操作建议、只有失败概括的文档，也存在文字提醒正确但学生仍未执行的情况。大量缺文档进一步削弱了预期干预覆盖率。</p>
<p><b>尚未证明：</b>特权信息是主要原因；某份训练 Skill 导致某道 dev 题失败；All-Skill 已发生不可逆能力遗忘；或简单增加训练步数/改为 Selective 一定能解决问题。当前只有一个训练 seed、每题 5 次评测，而且主分析集合按结果筛选。</p>
<p><b>后续若修订 Skill，优先清洗内容：</b>保留任务中精确的规格值；只使用当前观测允许的点击标签；明确“检索 → 打开商品 → 逐项选择 → 购买”的完成条件；删除“假设第一个商品匹配”的指令；把一次有效选择写入动作状态，选齐后结束。搜索策略可调整，但不能把搜索词出现规格误认为已经选中规格。此处仅给出修订建议，未改代码或恢复实验。</p>
<p class="note">dev 样本用于诊断和调参，不应再次充当独立最终评测。完整因果结论需要将改动与同预算、同起点的对照区分；当前用户已暂停实验，本报告不启动任何追加对照。</p>''')
parts.append('<h2>完整 46 题候选清单</h2>')
parts.append(table(['题号','初始 c/20','初始 pass@5','GRPO /5','All-Skill /5','分层'],[
 [esc(t[-5:]),f"{INDEX[t]['base_c']}/20",pct(INDEX[t]['base_pass5']),INDEX[t]['grpo_successes'],0,esc(LABEL[INDEX[t]['sampling_bin']])] for t in sorted(candidate,key=lambda t:(BINS.index(INDEX[t]['sampling_bin']),t))]))
parts.append('<div class="foot"><p><a href="analysis.json">结构化分析结果</a> · <a href="evidence.json">原始证据包（460 条候选轨迹、审计、来源哈希）</a> · <a href="related_training_skills.json">相关训练 Skill 与学生动作</a> · <a href="build_report.py">可重跑的本地分析脚本</a></p><p class="note">证据导出时间：'+esc(E['created_at'])+'。原始归档路径、文件哈希及抽样规则均保留。未拼接不同 checkpoint；未新增评测样本；未触碰暂停状态。</p></div></main></body></html>')
html='\n'.join(parts)
HTMLParser().feed(html)
for x in NOTES:
    assert html.count('id="case-'+x['id']+'"') == 1
(ROOT/'report.html').write_text(html)
print(json.dumps({k:derived[k] for k in ['overall','paired_outcomes','candidate_metrics','candidate_target_visits','sample_target_visits']},ensure_ascii=False,indent=2))
print('Wrote', ROOT/'report.html', 'bytes', len(html.encode()))
