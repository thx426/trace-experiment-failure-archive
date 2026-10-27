"""Read saved evidence; reproduce text compilation; write reports. No inference."""
import collections
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / 'evidence.json').read_text())
all_data = json.loads((HERE.parent / 'step100_comparison/action_audit.json').read_text())
namespace = {}
exec(compile(data['sources']['clean_skill.py'], 'saved_clean_skill.py', 'exec'), namespace)

NOTES = {
    'webshop_00510': '原稿推测 youth 或尺码可能不可得。指导组 sid0 实际已进入商品页，却发出无效的 click[size small]，随后购买。页面提供的是 small。score=0.9 仍不算成功。编译器保留精确选项及单项点击规则，但学生未遵守。研究员没有探索成功，也没有候选搜索词进入最终文档。',
    'webshop_00512': '唯一成功探索 ep005 的 guidance_hash=bare：这是研究员安排的一次无指导学生采样。最终搜索词来自这条轨迹。原稿把成功归于搜索词，未证明因果。指导组 sid1 进入正确商品页后直接购买，漏选颜色和尺寸。无指导 sid6、sid7 选项后购买成功。',
    'webshop_05104': '原始最终总结强调搜索词，并省略了成功轨迹中的两次选项点击。成功探索 ep008 实际先选 xx-large 和 multi 9，然后购买。编译文档保留这个精确规格和固定操作规则。指导组 sid0、sid1 按这一顺序成功；四条无指导轨迹均失败。不能据此区分模板与研究员探索各自的因果贡献。',
    'webshop_06866': '第二份探索原稿把 brown | beige 改成 brown or beige。最终原稿又把成功路径概括为 Search -> Buy Now，省略了尺寸、形状、颜色三次选择。编译器从公开题目保留完整 brown | beige，并要求逐项选择。成功探索和指导组 sid2 都有六步动作。指导组 sid0 进入另一商品页，分别点 brown 和 beige，继续尝试不存在的目标并购买失败。',
    'webshop_01418': '没有 finalize 原稿，不能把最后一次探索指导冒充最终 Skill。最终 origin=task_checklist_fallback。六次探索均失败。指导组 sid0、sid2 漏选后购买；无指导 sid6 已选 x-large 和 c-black，却继续重复、搜索，直到用完步数，没有购买。这一会话没有贡献经过成功轨迹支持的搜索方案。',
}

def instruction(case):
    return case['research_episodes'][0]['initial_obs'].split('Instruction: [SEP]', 1)[1].split('[SEP]', 1)[0].strip()

def show_trajectory(ep, observations=False):
    lines = [f"episode={ep.get('episode_id')} guidance_hash={ep.get('guidance_hash')} won={ep.get('won')} score={ep.get('score')} outcome={ep.get('outcome')}"]
    for step in ep['steps']:
        lines.append(f"  {step['t']:02d}. {step['action_norm']}  [valid={step['valid']}, acted={step['acted']}]")
        if observations:
            lines.append('      动作后观察（原文）: ' + step['observation'])
    return '\n'.join(lines)

docs = [item['docs']['doc.md'] for item in all_data['research_docs']]
assert len(docs) == 1600
common = set(docs[0].split('\n\n'))
for doc in docs[1:]:
    common.intersection_update(doc.split('\n\n'))
fixed_words = sum(len(p.split()) for p in common)
total_words = sum(len(doc.split()) for doc in docs)
query_sources = collections.Counter()
origins = collections.Counter()
for item, doc in zip(all_data['research_docs'], docs):
    audit = json.loads(item['meta']['self_assessment'])
    assert audit['free_form_prose_forwarded'] is False
    origins[audit['origin']] += 1
    if 'Observed in a successful attempt; verify current results.' in doc:
        query_sources['success_trajectory'] += 1
    elif 'Untested retrieval suggestion; verify current results.' in doc:
        query_sources['raw_proposal_untested'] += 1
    else:
        assert 'No checked search shortcut is supplied.' in doc
        query_sources['no_candidate_query'] += 1

verification = {
    'evidence_sha256': hashlib.sha256((HERE/'evidence.json').read_bytes()).hexdigest(),
    'global_doc_count': len(docs), 'global_fixed_paragraphs': len(common),
    'fixed_words_per_doc': fixed_words, 'total_doc_words': total_words,
    'constant_paragraph_word_share': fixed_words * len(docs) / total_words,
    'query_sources': dict(query_sources), 'origins': dict(origins),
    'all_free_form_prose_forwarded_false': True,
    'scope': 'CPU text compilation only. Removing final proposal retains all prior research choices and exploration records. Not a training ablation.',
    'cases': [],
}
overview = [
    'Clean-Skill v2：五道真实训练题的研究员原稿、编译文档与学生动作对照',
    '证据导出时间：' + str(data['export_time']),
    '范围：只读现有实验记录，另做 CPU 文本重编译。没有模型推理，没有启动实验，没有修改冻结训练代码。',
    '样本选择：按机制选取五个训练会话，覆盖成功、失败、原稿错误和检查表兜底。不是随机样本，也不是第100步 DEV 评测。',
    '同组八条 rollout 使用同一个学生策略版本。sid0–3 属于指导组，sid4–7 属于无指导组。已逐条核实指导组 steer_count>0、无指导组为0。steer_count 不是环境动作步数。',
    '成功按 won 统计。部分分数 score<1 不算成功。四条样本不足以作稳定的成功率或因果估计。',
    '',
    '主要结论：当前 Skill 主要是人工编写的固定操作规则，加上公开题目规格，以及从探索轨迹机械提取的搜索词和失败提醒。研究员最终自由文本没有直接转交学生。',
    f'全部1600份文档有{len(common)}段完全相同的文字，每份共{fixed_words}个空白分隔词，占总词数{fixed_words*len(docs)/total_words:.4%}。这不是信息贡献或因果收益占比。其余文字仍含其他固定可选提示。',
    '最终候选搜索词来源：' + json.dumps(dict(query_sources), ensure_ascii=False),
    '文档产生方式：' + json.dumps(dict(origins), ensure_ascii=False),
    '编译器不仅处理最终原稿。研究员 run_experiment 的非空 guidance 也先经过同一编译器。空 guidance 对应无指导探索。',
    '研究员贡献：选择探索指导、调用时机和采样次数；安排有指导或无指导试跑；查看轨迹并安排后续探索；提交候选查询。探索轨迹中的搜索词由学生实际生成。',
    '编译器贡献：从公开题目提取精确规格；写入固定操作流程；优先提取成功轨迹中的合规搜索词；按无效动作和重复动作计数加入固定提醒。其词汇检查不是语义正确性的完整证明。',
    '全部文档中，只有25份最终候选查询来自原稿中的 search[...]，并标为未验证。其他自由解释、猜测和动作路径不进入最终文档。',
    '',
    '五个训练会话（指导成功 / 无指导成功）：',
]

for c in data['cases']:
    task = instruction(c)
    final_calls = [x for x in c['raw_skill_calls'] if x['name'] == 'finalize']
    assert len(final_calls) <= 1
    proposal = final_calls[0]['arguments']['experience_doc'] if final_calls else ''
    records = [SimpleNamespace(trajectory=SimpleNamespace(won=ep['won'], steps=[SimpleNamespace(**s) for s in ep['steps']])) for ep in c['research_episodes']]
    meta_audit = json.loads(c['meta']['self_assessment'])
    compiled, actual_audit = namespace['compile_skill']('Instruction: ' + task, records, proposal, meta_audit['origin'])
    no_proposal, _ = namespace['compile_skill']('Instruction: ' + task, records, '', meta_audit['origin'])
    assert compiled == c['compiled_skill'], c['task_id']
    assert no_proposal == compiled, c['task_id']
    assert actual_audit == meta_audit, c['task_id']
    assert len(c['student_rollouts']) == 8
    guided = []; bare = []
    for r in c['student_rollouts']:
        assert r['policy_version'] == {'min_global_steps': c['step']-1, 'max_global_steps': c['step']-1}
        assert c['archive'].endswith(r['research']['research_archive'])
        is_guided = r['steering']['steer_arm'] == 1
        assert (r['steering']['steer_count'] > 0) == is_guided
        assert is_guided == (r['session_id'] < 4)
        (guided if is_guided else bare).append(r)
    row = {'task_id': c['task_id'], 'step': c['step'], 'policy_version': c['step']-1,
           'research_episodes': len(records), 'research_successes': sum(ep['won'] for ep in c['research_episodes']),
           'raw_finalize_exists': bool(final_calls), 'origin': meta_audit['origin'],
           'compiled_reproduced': True, 'without_final_proposal_identical': True,
           'guided_successes': sum(r['won'] for r in guided), 'bare_successes': sum(r['won'] for r in bare),
           'candidate_query': meta_audit['candidate_query'], 'research_wall_s': c['meta']['research_wall_s']}
    verification['cases'].append(row)
    overview += [f"{c['task_id']}：训练step{c['step']}，学生policy{c['step']-1}；探索成功{row['research_successes']}/{len(records)}；指导{row['guided_successes']}/4，无指导{row['bare_successes']}/4。", NOTES[c['task_id']], '完整原稿、编译稿、全部动作：' + str(HERE/(c['task_id']+'.txt')), '']
    lines = [c['task_id'] + ' 完整证据对照', '训练步：'+str(c['step'])+'；八条训练rollout的学生policy版本：'+str(c['step']-1),
             '选取性质：机制案例，非随机样本，非最终DEV评测。', '公开任务原文：', task,
             '', '证据结论：', NOTES[c['task_id']], '',
             '一、研究员最终原始 Skill（finalize.experience_doc 原文）',
             proposal if final_calls else '无。该会话没有 finalize 调用。后续编译文档由 task_checklist_fallback 产生。',
             '', '二、研究员每次探索提交的原始指导（原文）']
    for call in c['raw_skill_calls']:
        if call['name'] != 'run_experiment':
            continue
        lines += [f"session.jsonl 第{call['line']}行，run_experiment 参数：", json.dumps(call['arguments'], ensure_ascii=False, indent=2)]
    lines += ['', '三、最终编译 Skill（doc.md 原文）', c['compiled_skill'], '',
              '编译审计：', json.dumps(meta_audit, ensure_ascii=False, indent=2),
              'CPU复核：保存的原稿、公开任务和探索记录可精确重现doc.md；只删除最终原稿，仍得到完全相同的文档。此操作保留了研究员先前安排的全部探索，因此不能视为去研究员的训练消融。',
              '', '四、研究员探索的学生动作（全部探索轨迹）']
    for ep in c['research_episodes']:
        lines += [show_trajectory(ep), '']
    lines += ['五、正式训练的八条学生动作（全部）']
    for r in c['student_rollouts']:
        arm = '指导组' if r['steering']['steer_arm'] else '无指导组'
        lines += [f"sid{r['session_id']} {arm}，steer_count={r['steering']['steer_count']}，won={r['won']}，score={r['score']}",
                  show_trajectory(r['trajectory']), '']
    lines += ['六、探索时实际使用的指导（非空原稿也经过编译；以下为guidance_log原文）']
    for g in c['meta']['guidance_log']:
        lines += [f"experiment_idx={g['experiment_idx']} k={g['k']} guidance_hash={g['guidance_hash']} episodes_before={g['episodes_before']}", g['guidance'], '']
    lines += ['七、正式训练动作后的页面观察（用于检查可用目标、选择是否有效）']
    for r in c['student_rollouts']:
        lines += [f"sid{r['session_id']} 初始观察：", r['trajectory']['initial_obs'], show_trajectory(r['trajectory'], observations=True), '']
    lines += ['八、来源', '远端会话目录：'+c['archive'], '原始文件SHA256：', json.dumps(c['source_sha256'], indent=2), '训练rollout来源及SHA256：']
    lines += [r['file']+'\n'+r['sha256'] for r in c['student_rollouts']]
    (HERE/(c['task_id']+'.txt')).write_text('\n\n'.join(lines), encoding='utf-8')

overview += ['离线复核：四份已有最终原稿删掉后，编译结果均不变。第五题本来没有最终原稿。五份保存的文档全部精确复现。',
             '边界：不能据此断言研究员没有价值，也不能把Clean-Skill的总收益归于研究员。要分离贡献，需要保持预算可比的检查表基线；本次没有启动这种实验。',
             '完整机器可读证据：'+str(HERE/'evidence.json'), '机器可读复核：'+str(HERE/'offline_verification.json')]
(HERE/'overview.txt').write_text('\n\n'.join(overview), encoding='utf-8')
(HERE/'offline_verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(verification, ensure_ascii=False, indent=2))
