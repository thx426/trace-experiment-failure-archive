"""Build a reproducible offline review from already saved data; no model calls."""
from pathlib import Path
import json,collections,re,difflib,statistics,hashlib
P=Path(__file__).resolve().parent
d=json.loads((P/'action_audit.json').read_text())
s=json.loads((P/'derived_final_stats.json').read_text())
cmp=json.loads((P/'complete_audited_per_task.json').read_text())
g=json.loads((P.parents[1]/'pass5_20261023/grpo_final_review/report.json').read_text())['gate']
notes={
'00985':('精确规格替换','5次都找到正确商品，却将 12 pack tank top undershirts 选成 12 pack mix；不是检索失败。','在购买前逐字核对颜色/套装规格；不能把共享商品的另一变体作为默认值。','06746'),
'01705':('漏选属性','5次都找到正确地毯，选尺寸→颜色→重复尺寸→购买，均漏掉 item shape: runner。','用按属性名分开的已选状态表；补 click[runner]，无需重复尺寸。','04233'),
'02397':('检索与商品选择偏移','5次均打开 Marvel T-shirt（b07xpr3r7n），GRPO成功轨迹打开 Tigger Tank Top（b08226ndzw）；选项都有效但最终得分为0。Clean查询省略多项材质/版型特征。','保留区分商品的特征词，核验商品类型/特征，不能仅凭颜色、fit type和尺码相同就购买。','03262'),
'02839':('检索偏移与错误替代','4次打开Nautica商品，click[5x-large]无效后选择nautica red并购买；另1次打开内衣套装并选择12 pack mix。','5x-large与a-red必须同时可选；不能用nautica red替代a-red。当前候选不满足时返回检索。','02909'),
'03385':('检索偏移与错误替代','5次选择错误泳装b09s6lzyhq并选择pink/yellow；要求green。GRPO成功用tummy control、high waist检索另一商品。','保留tummy control/high waist；找不到green时换候选，不能选择别的颜色后下单。','04359'),
'03571':('检索偏移与无效动作后购买','5次查询仍保留158- purple，却省略slim fit，均打开b09qqp3356；紫色点击无效后仍购买。GRPO成功商品是b09r9ycm6r。','保留slim fit；颜色目标不可用即判定当前候选不合格，禁止继续购买。','04270'),
'04195':('选项被后续动作覆盖','5次均在同一商品先选black，再选white，最后购买；GRPO仅1次重新选black后成功。任务含white item，但明确color:black。','区分描述性特征和明确选项字段；最终颜色必须为black，不能被white覆盖。',''),
'04717':('检索与商品选择偏移','5次查询省略long sleeve/unique design，选择b08g8dxr5n，GRPO选择b09krlgsc5；颜色/尺码都选了但得分仅0.7143。','不要把选项齐全当作商品符合全部条件；保留特征词并核验候选。',''),
'05317':('检索偏移与无效动作后购买','5次没有选到GRPO成功的b01hqtwl6s；4次尝试不存在的red，另1次选xnj-tshirt328-red；均购买。','保留wash cold、dry clean、tumble dry等区分词；核对精确red选项与商品属性，不满足则换候选。','05837'),
'06752':('检索偏移与规格近似替代','5次查询丢掉long sleeve/big tall，选Nautica b07hrfsnl4，将4x-large big tall选成4x-large tall，black选成true black。','保留long sleeve和big tall；精确规格不做同义近似替换，不满足时重新找商品。','00932'),
}
train={x['task_id']:x['extra_info']['instruction'] for x in d['datasets']['clean']['train']}
dev={x['task_id']:x['extra_info']['instruction'] for x in d['datasets']['clean']['dev']}
def stem(q):return re.sub(r'\s+',' ',q.rsplit(' with ',1)[0].lower())
def action_view(r):
    return {k:r.get(k) for k in ['file','task_id','session_id','sample_id','won','score','outcome','truncated','evaluation_bare']} | {'steps':[{'t':x['t'],'action':x['action_norm'],'valid':x['valid'],'acted':x['acted'],'observation':x['observation']} for x in r['trajectory']['steps']]}
losses=[]
for x in s['lost']:
    tid=x['task_id'];kind,cause,correction,near=notes[tid[-5:]]
    z=x|{'failure_type':kind,'observed_failure':cause,'suggested_correction_not_executed':correction,'eval_skill_present':False,'skill_relationship':'dev题不在训练/研究员任务集合中。以下训练文档仅作相似任务的机制参照，不是本题接受的Skill，也不是因果证据。','trajectories':{a:[action_view(r) for r in sorted(d['trajectories'][a],key=lambda r:r['session_id']) if r['task_id']==tid] for a in ['grpo','clean']}}
    if near:
        nt='webshop_'+near;doc=max((r for r in d['research_docs'] if r['meta']['task_id']==nt),key=lambda r:r['meta']['global_step'])
        z['related_training_skill']={'task_id':nt,'instruction':train[nt],'global_step':doc['meta']['global_step'],'source':doc['path'],'assessment':json.loads(doc['meta']['self_assessment']),'document':doc['docs']['doc.md']}
    losses.append(z)
behaviour={}
for arm in ['grpo','clean']:
    result={}
    for label,rows in [('all',d['trajectories'][arm]),('lost10',[r for r in d['trajectories'][arm] if r['task_id'] in {z['task_id'] for z in losses}])]:
        c=collections.Counter();qlen=[];alen=[]
        for r in rows:
            st=r['trajectory']['steps'];acts=[x['action_norm'] for x in st];alen.append(len(acts))
            qlen += [len(a[7:-1].split()) for a in acts if a.startswith('search[')]
            c['purchased']+=int('click[buy now]' in acts)
            c['invalid_action_episode']+=int(any(not x['valid'] for x in st))
            c['opened_description_or_features']+=int(any(a in ['click[features]','click[description]'] for a in acts))
            c['truncated']+=int(r.get('truncated',False))
            products=[a[6:-1] for a in acts if re.fullmatch(r'click\[b[0-9a-z]{9}\]',a)]
            hits=re.findall(r'\bB[0-9A-Z]{9}\b',st[0]['observation']) if st else []
            c['first_opened_is_first_search_result']+=int(bool(hits and products and hits[0].lower()==products[0]))
        result[label]={'episodes':len(rows),**dict(c),'mean_action_count':statistics.mean(alen),'mean_search_query_words':statistics.mean(qlen)}
    behaviour[arm]=result
audit={'task_id_overlap':sorted(set(train)&set(dev)),'identical_full_instruction_pairs':[(a,b) for a,q in dev.items() for b,v in train.items() if q==v],'dev_with_shared_pre_option_prefix':sum(stem(q) in set(map(stem,train.values())) for q in dev.values()),'pre_option_prefix_definition':'去掉最后一个 with 之后的选项规格段，压缩空白并转小写；仅为文字模板相似性，不等于相同商品。','research_dev_overlap':sorted({r['meta']['task_id'] for r in d['research_docs']}&set(dev)),'same_datasets_between_arms':d['datasets']['clean']==d['datasets']['grpo'],'train_config_differences':{k:[d['configs']['grpo']['train'].get(k),v] for k,v in d['configs']['clean']['train'].items() if d['configs']['grpo']['train'].get(k)!=v},'frozen_manifest_changes':d['manifest_changed']}
assess=[json.loads(r['meta']['self_assessment']) for r in d['research_docs']]
audit['research_session_outcomes']={'sessions':len(assess),'with_successful_exploration':sum(r['successes']>0 for r in assess),'without_successful_exploration':sum(r['successes']==0 for r in assess),'candidate_queries':sum(bool(r['candidate_query']) for r in assess),'researcher_finalized':sum(r['origin']=='researcher_finalized' for r in assess),'task_checklist_fallback':sum(r['origin']=='task_checklist_fallback' for r in assess)}
out={'source_export_time':d['time'],'evaluation_audits':cmp['audits'],'summary':s,'comparability':audit,'behaviour':behaviour,'lost_task_analysis':losses,'all_tasks':[{'task_id':r['task_id'],'instruction':r['instruction'],'initial_gate':g[r['task_id']],**{f'{a}_{k}_k':r[f'{a}_{k}_k'] for a in ['grpo','clean'] for k in [20,40,60,80,100]},'lost_at20':r['lost_at20']} for r in cmp['rows']]}
if (P/'learning_signal_summary.json').exists():out['training_signal']=json.loads((P/'learning_signal_summary.json').read_text())
(P/'final_review_evidence.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
lines=['Clean-Skill v2 与普通 GRPO：第100步最终审计与逐题分析','仅分析已保存的数据；未启动新实验、未改模型或训练配置。',
'评测口径：同一256道dev题，每题5次，temperature=0.6。pass@1估计=1280次成功比例；pass@5=256题中至少成功一次的比例。',
'所有第100步记录已核验同一launch、step100权重、bare=true、空steering/research。两组都是Qwen2.5-3B学生无Skill评测。',
'GRPO：937/1280=73.203125%；205/256=80.078125%。Clean：1019/1280=79.609375%；214/256=83.59375%。差值+6.40625/+3.515625个百分点。',
'逐题成功次数：58提升、18下降、180相同。19题从0/5变为有成功；10题从有成功变0/5。5/5题从158变193。',
'第20步丢失的28题全部恢复，其中27题5/5；最终10道丢失题与这28题不重合。',
'净增加82次成功 = 新解题贡献74 − 丢失题损失39 + 双方都能做对的题净增加47。',
'同题配对bootstrap（20000次，仅评测题抽样不确定性）：pass@1增益95%区间[2.42,10.39]pp；pass@5[-0.39,7.81]pp。pass@5精确McNemar双侧p=0.136。不是多训练seed置信区间。',
'', '你实际改变了什么',
'保持同一初始学生、seed42、相同训练题目/日程、16题×8条rollout、100步、lr=1e-6、二值成功奖励、普通GRPO。',
'Clean每题4条指定guided、4条bare；研究员探索后把指导编译为精确公开规格+逐项点击+满足条件后购买+失败恢复检查表，只保留受约束的检索建议。自由文本不直接注入。',
'改动还包括收尾轮次保留、探索预算结束后限制重复读取、明确标注检查表兜底、缺文档/基础设施错误中止、独立规格审计。',
'guided臂使用Skill影响采样，和bare臂一起做GRPO；当前不是OPD/SFT，也不是历史asymmetric PGS目标。adv归一化关闭；w_beh_mode=off，不能宣称已完整消除引导采样偏差。',
'没有做A/B/C/D，因此无法把增益唯一归因于某一条清洗规则。训练结束后的bare提升支持部分迁移，不代表全部指导都被学会。',
'', '为什么这次分数高，解释到哪里为止',
'可观察改善：无效动作episode从183降至127，购买完成从1256升至1278，五次全对题增加35；查询平均词数从15.36降到6.64。',
'查询简化有双面性：00511、00989、01102、01987等更易找对商品；02397、03571、04717、06752等删掉区分词后找错。不能说查询越短越好。',
'两组均没有打开Description/Features；首次选中搜索首项分别1251/1280和1273/1280。这是一套在固定WebShop分布上更有效的搜索/选择/购买策略，还不是充分的语义核验或多候选恢复策略。',
'训练与dev题ID和完整指令无重合；212/256 dev题的规格前文字前缀出现在训练中。这是同域/相近模板迁移，不能据此声称未知商品或开放场景泛化。',
'研究员额外成本：1600研究会话、3502探索实验、7511额外episode；学生正式训练两组均12800episode。不是等总计算预算的胜利。',
'评测用dev，并已用于人工分析与停止规则；单seed。不是独立保留test的最终证据。传输分片/幂等修复保持奖励和环境语义，但可能改变时序和随机采样顺序。',
'', '按初始无Skill pass@5分层（冻结gate：每题20次，1-C(20-c,5)/C(20,5)；不是把第100步k/5称为pass@5）',
'层\t题数\tGRPO pass@1\tClean pass@1\tGRPO pass@5\tClean pass@5\t新增/丢失']
for level,v in s['strata'].items():
    lines.append(f"{level}\t{v['n']}\t{v['grpo']['pass1']:.2%}\t{v['clean']['pass1']:.2%}\t{v['grpo']['pass5']:.2%}\t{v['clean']['pass5']:.2%}\t{v['new']}/{v['lost']}")
lines+=['低层<0.5；中层[0.5,0.8)；高层≥0.8。10道丢失题：7道初始为0、2道44.74%、1道87.09%；中层无丢失。', '', '新增成功的19道题（GRPO100 → Clean100成功次数/5）']
lines += [f"{r['task_id']}\t0/5 → {r['clean_100_k']}/5\t{r['instruction']}" for r in s['newly_solved']]
lines+=['', '10道完全丢失题：全部50次Clean失败都有购买，无截断，也不是环境超时。7题主要检索/商品选择错误，3题为正确商品上的规格错误。']
for z in losses:
    lines += ['',f"{z['task_id']} | 初始pass@5 {z['initial_gate']['pass_at_5']:.2%}（{z['initial_gate']['c']}/20） | GRPO {z['grpo_100_k']}/5 → Clean 0/5",z['instruction'],z['observed_failure'],'纠正建议（未执行）：'+z['suggested_correction_not_executed'],z['skill_relationship']]
    if 'related_training_skill' in z:
        q=z['related_training_skill'];a=q['assessment'];lines += [f"相似训练任务：{q['task_id']}，step {q['global_step']}；明确规格 {json.dumps(a['required_options'],ensure_ascii=False)}；候选查询 {a['candidate_query']}；探索成功数 {a['successes']}。"]
    lines += ['动作证据（不包含模型内部思维文本；✓有效，×无效）：']
    for arm in ['grpo','clean']:
        for r in z['trajectories'][arm]:
            lines.append(f"  {arm} sid={r['session_id']} won={r['won']} score={r['score']:.6f} [{r['file']}] "+' → '.join(('✓' if t['valid'] else '×')+t['action'] for t in r['steps']))
lines+=['', '另外8道成功率下降但未完全丢失的题']
lines += [f"{r['task_id']}\t{r['grpo_100_k']}/5 → {r['clean_100_k']}/5\t{r['instruction']}" for r in s['regressed'] if r['clean_100_k']>0]
if 'training_signal' in out:
    ts=out['training_signal']
    lines+=['','训练奖励差异与优势代理（不是读取实际优势张量）',
      '两组每一步训练题集合完全相同，reward_from_score=False，使用won二值奖励。不能把连续环境score的差异直接当作训练优势。',
      '1—20步：Clean组内二值奖励不全相同127/320=39.69%，GRPO116/320=36.25%。全程则为543/1600=33.94%对776/1600=48.50%。',
      '全程Clean全成功组633，GRPO374；全失败组424对450。非零优势组比例下降可能来自成功饱和，不能单独作为方法好坏指标。',
      'Clean全程81组引导臂至少1次成功而裸跑4次全失败；反方向73组。两臂总成功数3681/6400对3644/6400，差距很小，不能声称引导臂全面压倒裸跑。',
      '前20步存在18个同题同步组：GRPO8次全失败，Clean裸跑4次全失败而引导有成功。这里只说明改变探索分布确实产生了有价值样本，不隔离训练随机性或单个清洗组件的因果作用。',
      '实例：step2 / webshop_05104，Clean奖励[1,1,0,0,0,0,0,0]，前4条为引导臂；按已核验普通GRPO且不除std的公式，重构优势为[0.75,0.75,-0.25,-0.25,-0.25,-0.25,-0.25,-0.25]。同题同步GRPO为8个0。此处是公式重构，不冒充实际训练优势张量。',
      json.dumps({'windows':ts['windows'],'first_three_examples':ts['examples'][:3]},ensure_ascii=False,indent=2)]
lines+=['','完整256题对照（k=5次中的成功次数；初始pass@5来自20次冻结基线）','task_id\t初始pass@5\t初始层\tGRPO100 k\tClean100 k\t变化\tGRPO20 k\tClean20 k\t任务']
for r in sorted(cmp['rows'],key=lambda x:x['task_id']):
    tid=r['task_id'];delta=r['clean_100_k']-r['grpo_100_k'];lines.append(f"{tid}\t{g[tid]['pass_at_5']:.6f}\t{g[tid]['stratum']}\t{r['grpo_100_k']}\t{r['clean_100_k']}\t{delta:+d}\t{r['grpo_20_k']}\t{r['clean_20_k']}\t{r['instruction']}")
lines+=['','证据文件：complete_audited_per_task.json；summary_complete.json；action_audit.json；final_review_evidence.json；derived_final_stats.json。','审计仅复核当前保存的训练/评测证据；因果归因需独立消融，但本次没有启动任何新实验。']
(P/'final_review_zh.txt').write_text('\n'.join(lines)+'\n')
print('wrote',P/'final_review_zh.txt', 'losses',len(losses),'all_tasks',len(cmp['rows']))
