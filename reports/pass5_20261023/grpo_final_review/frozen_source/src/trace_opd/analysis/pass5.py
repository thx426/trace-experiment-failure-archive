"""Frozen task strata from independent untrained no-Skill rollouts."""
from collections import defaultdict
import hashlib,json,math
from pathlib import Path
VERSION='base_pass5_v1'
def pass_at_k(n,c,k=5):
 if any(type(x) is not int for x in (n,c,k)) or not (1<=k<=n and 0<=c<=n):raise ValueError('Invalid counts')
 return 1-math.comb(n-c,k)/math.comb(n,k) if n-c>=k else 1.0
def stratum(p):return 'low' if p<.5 else 'middle' if p<.8 else 'high'
def build_table(rows,expected_tasks,*,n,student):
 if n<20 or not expected_tasks or not student:raise ValueError('Missing inventory or insufficient replicates')
 by_task=defaultdict(dict)
 for row in rows:
  tid=row['task_id'];idx=row['sample_idx']
  if tid not in expected_tasks or row['split']!=expected_tasks[tid]:raise ValueError('Unexpected task/split '+tid)
  format_failure = row.get('outcome')=='DEGENERATE:format' and row.get('error')=='no parseable action repeatedly' and row.get('won') is False
  if row.get('arm')!='bare' or (row.get('error') and not format_failure) or row.get('outcome','').lower() in ('env_error','error'):raise ValueError('Invalid no-Skill attempt '+tid)
  if type(idx) is not int or not 0<=idx<n or idx in by_task[tid]:raise ValueError('Duplicate/invalid replicate '+tid)
  if type(row.get('won')) is not bool:raise ValueError('Invalid success field '+tid)
  by_task[tid][idx]=row['won']
 tasks={}
 for tid,split in sorted(expected_tasks.items()):
  if split not in ('train','dev') or set(by_task[tid])!=set(range(n)):raise ValueError('Incomplete probe '+tid)
  c=sum(by_task[tid].values());p=pass_at_k(n,c)
  tasks[tid]={'split':split,'n':n,'c':c,'pass_at_5':p,'stratum':stratum(p),'skill_eligible':p<.8}
 return {'version':VERSION,'source_arm':'bare','student':student,'k':5,'n':n,'threshold':.8,'tasks':tasks}
def load_table(path,*,student):
 t=json.loads(Path(path).read_text())
 if t.get('version')!=VERSION or t.get('source_arm')!='bare' or t.get('student')!=student or t.get('k')!=5 or t.get('threshold')!=.8 or t.get('n',0)<20:raise ValueError('Incompatible gate provenance')
 tasks=t.get('tasks',{})
 if not tasks:raise ValueError('Empty gate')
 for tid,r in tasks.items():
  p=pass_at_k(r['n'],r['c'])
  if r['n']!=t['n'] or r['split'] not in ('train','dev') or not math.isclose(r['pass_at_5'],p,abs_tol=1e-12) or r['stratum']!=stratum(p) or type(r['skill_eligible']) is not bool or r['skill_eligible']!=(p<t['threshold']):raise ValueError('Invalid gate '+tid)
 return tasks
def training_decision(table,task_id):
 if not table:return {'task_gate_applied':0,'task_skill_eligible':1,'base_pass5':-1.0}
 if task_id not in table or table[task_id]['split']!='train':raise ValueError('Unknown training task '+task_id)
 r=table[task_id];return {'task_gate_applied':1,'task_skill_eligible':int(r['skill_eligible']),'base_pass5':r['pass_at_5']}
def file_sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while b:=f.read(4*1024*1024):h.update(b)
 return h.hexdigest()
