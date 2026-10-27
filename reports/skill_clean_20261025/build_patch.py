from pathlib import Path
import json
R=Path(__file__).resolve().parent
F=R/'frozen'
P=R/'patch'
def emit(rel,s):
 p=P/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s);compile(s,rel,'exec')
def replace(s,a,b):
 assert s.count(a)==1,(a[:90],s.count(a))
 return s.replace(a,b)

emit('src/trace_opd/research/clean_skill.py',(R/'clean_skill.py').read_text())
s=(F/'src/trace_opd/research/compact.py').read_text()
a=s.index('Finalize when sufficient evidence')
b=s.index('"""',a)
s=s[:a]+'''Finalize when the experiment budget is spent. At most two evidence fetches are
allowed afterwards; finalization is mandatory when FINALIZE NOW is present.
Submit a concise 120-180 word diagnosis and conditional action suggestions.
Keep exact task option strings; never treat codes as typos or omit constraints.
Search words do not select options. Open a product, select all required options
using current clickables, then buy immediately when configured. Record prior
valid selections rather than repeat unchanged page actions. Never assume the
first result satisfies the instruction. Distinguish failures from established
success. Your free-form draft is archived; a task-grounded compiler keeps exact
specifications and checked search suggestions for the executor. Do not write
product identifiers or brands discovered privately as standalone answers.
''' +s[b:]
s=replace(s,'self.last_result = "Begin with a useful experiment."','self.last_result = "Begin with a useful experiment."\n        self.fetch_after_exhausted = 0\n        self.finalize_only = False')
s=replace(s,'self.last_result = result\n','self.last_result = result\n        if call.name == "fetch_raw" and self.router.state.experiments_used >= self.cfg.n_exp:\n            self.fetch_after_exhausted += 1\n')
s=replace(s,'offered = schemas(self.cfg, self.router.state)', '''self.finalize_only = (turn >= self.max_turns - 2 or
                              (self.router.state.experiments_used >= self.cfg.n_exp and
                               self.fetch_after_exhausted >= 2))
        offered = schemas(self.cfg, self.router.state)
        if self.finalize_only:
            offered = [t for t in offered if t['function']['name'] == 'finalize']''')
s=replace(s,'body = self.state_text(turn, detail)', '''body = self.state_text(turn, detail)
            if self.finalize_only:
                body += "\\nFINALIZE NOW. Only finalize is available. No additional exploration or raw fetch. Submit a concise draft in this reply."''')
s=replace(s,'evidence = rec.narration.text if rec.narration and rec.narration.text else rec.mechanical_log', '''# Mechanical actions take precedence over lossy/hallucinated narration.
            evidence = "ACTIONS: " + "; ".join(
                f"{x.t}:{x.action_norm} valid={x.valid}" for x in t.steps)
            evidence += f"; success={bool(t.won)}; score={t.score}"''')
emit('src/trace_opd/research/compact.py',s)

s=(F/'src/trace_opd/research/tools.py').read_text()
s=replace(s,'self._seen_guidance_hashes: set[str] = set()', '''self._seen_guidance_hashes: set[str] = set()
        self._clean_skill = research.context_mode == "bounded" and adapter.task_id.startswith("webshop_")''')
s=replace(s,'            if call.name == "run_experiment":', '''            if getattr(getattr(self, "_compact_context", None), "finalize_only", False) and call.name != "finalize":
                return "FINALIZE NOW: exploration/read budget is closed; call finalize."
            if call.name == "run_experiment":''')
s=replace(s,'        guidance = str(args.get("guidance") or "")', '''        guidance = str(args.get("guidance") or "")
        if guidance.strip() and self._clean_skill:
            from .clean_skill import compile_skill
            guidance, _audit = compile_skill(self.adapter.task_brief(), st.episodes.values(), guidance, "exploration")''')
s=replace(s,'        violations = self._check_doc(doc)', '''        if self._clean_skill:
            from .clean_skill import finalize_clean
            return finalize_clean(self, doc, self_assessment)
        violations = self._check_doc(doc)''')
emit('src/trace_opd/research/tools.py',s)

s=(F/'src/trace_opd/research/session.py').read_text()
s=replace(s,'        return self.snapshot()\n', '''        if self._compact is not None and getattr(self._router, "_clean_skill", False) and not state.finalized and state.episodes:
            # Normal turn/output exhaustion: emit a labelled public-task checklist,
            # never turn an infrastructure error into a fabricated research success.
            from .clean_skill import finalize_clean
            finalize_clean(self._router, origin="task_checklist_fallback")
        return self.snapshot()
''')
emit('src/trace_opd/research/session.py',s)

s=(F/'src/trace_opd/train/agent_loop.py').read_text()
s=replace(s,'        attempts = result.last_exp_attempts if result is not None else 0', '''        if rcfg.context_mode == "bounded" and adapter.task_id.startswith("webshop_") and not doc:
            from ..research.compact import contract_error
            contract_error(f"clean Skill absent for {adapter.task_id}; archive={archive_path}; error={error}")
        attempts = result.last_exp_attempts if result is not None else 0''')
emit('src/trace_opd/train/agent_loop.py',s)

# Preserve old run unchanged. A fresh name and explicit provenance distinguish
# this guidance-method change from the earlier compact infrastructure repair.
s=(F/'run_sequence.py').read_text().replace('24r3','25clean1')
s=replace(s,"for name in ['all_skill','selective_skill']:","for name in ['clean_skill']:")
s=s.replace("['all_skill','selective_skill']","['clean_skill']")
s=replace(s,"OWNERS={'all_skill':'/tmp/p5compact3072-all25clean1','selective_skill':'/tmp/p5compact3072-select25clean1'}", "OWNERS={'clean_skill':'/tmp/p5compact3072-clean25v1'}") if "OWNERS={'all_skill':'/tmp/p5compact3072-all25clean1','selective_skill':'/tmp/p5compact3072-select25clean1'}" in s else replace(s,"OWNERS={'all_skill':'/tmp/p5compact3072-all25clean1','selective_skill':'/tmp/p5compact3072-selective25clean1'}", "OWNERS={'clean_skill':'/tmp/p5compact3072-clean25v1'}")
s=s.replace("R/'configs/all_skill.yaml'","R/'configs/clean_skill.yaml'")
s=s.replace("next_job='selective_skill' if name=='all_skill' else None","next_job=None")
s=replace(s,'            inspect_service_log()\n            time.sleep(5)', '''            inspect_service_log()
            from baseline_guard import review_completed_evaluations
            review_completed_evaluations(R, run, cfg)
            time.sleep(5)''')
s=replace(s,"    status('stopped_on_error',error=repr(exc));traceback.print_exc()", '''    if (R/'baseline_stop.json').exists():
        status('stopped_below_baseline', comparison=json.loads((R/'baseline_stop.json').read_text()))
    else:
        status('stopped_on_error',error=repr(exc));traceback.print_exc()''')
s=replace(s,"files", "files") if False else s
s=s.replace('Approved formal sequence: compact All-Skill -> compact Selective-Skill.','Approved formal single run: Clean-Skill GRPO; no old All/Selective queue.')
(R/'run_sequence.py').write_text(s);compile(s,'run_sequence.py','exec')
print('PATCH_READY',len(list(P.rglob('*.py'))))
