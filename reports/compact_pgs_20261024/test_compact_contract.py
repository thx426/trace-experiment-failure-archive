"""CPU-only contract tests. No model inference or environment episodes."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest

R=Path(__file__).resolve().parent
modfile=R/'code/src/trace_opd/research/compact.py'
if not modfile.exists():
    modfile=R/'patch/src/trace_opd/research/compact.py'
spec=importlib.util.spec_from_file_location('compact_contract',modfile)
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


class CharTokenizer:
    def encode(self, text, **kwargs):
        return NS(ids=list(map(ord,text)))
    def decode(self, ids, **kwargs):
        return ''.join(map(chr,ids))


def context():
    obj=c.CompactContext.__new__(c.CompactContext)
    obj.cfg=NS(n_exp=3,k_default=2,k_max=4,fetch_raw_limit=20,doc_max_words=500,
               request_token_limit=3072,context_target_tokens=2500,raw_page_tokens=512)
    obj.max_turns=16;obj.task='TASK_SENTINEL: buy the requested item'
    state=NS(experiments_used=0,revision_requested=False,episodes={},fetch_used=0,
             exp_successes_total=0,exp_attempts_total=0,guidance_log=[])
    obj.router=NS(state=state);obj.tokenizer=CharTokenizer()
    obj.last_call=None;obj.last_result='Begin.'
    return obj


class ContractTests(unittest.TestCase):
    def test_exhausted_tool_and_refund(self):
        x=context();s=x.router.state
        names=lambda:[t['function']['name'] for t in c.schemas(x.cfg,s)]
        self.assertIn('run_experiment',names())
        s.experiments_used=3;self.assertNotIn('run_experiment',names())
        s.experiments_used=2;self.assertIn('run_experiment',names())
        s.revision_requested=True;self.assertEqual(names(),['finalize'])

    def test_long_history_bounded_and_task_retained(self):
        x=context();s=x.router.state
        s.guidance_log=[dict(experiment_idx=i,guidance_hash=f'g{i}',k=2,guidance='x'*20000) for i in range(3)]
        x.last_call=NS(name='run_experiment');x.last_result='evidence'*10000
        async def count(messages,tools):
            # Include schemas and template overhead; each byte is a mock token.
            return len(json.dumps([messages,tools]))
        # A realistic tokenizer uses multiple characters per token.
        async def realish(messages,tools):return (await count(messages,tools))//2
        messages,tools,n=asyncio.run(x.build(4,realish))
        self.assertLessEqual(n,3072)
        self.assertIn('TASK_SENTINEL',messages[1]['content'])
        self.assertNotIn('x'*1000,messages[1]['content'])
        self.assertEqual(len(s.guidance_log[0]['guidance']),20000)

    def test_oversize_mandatory_content_fails_closed(self):
        x=context();x.task='t'*20000
        async def count(messages,tools):return sum(len(m['content']) for m in messages)
        with tempfile.TemporaryDirectory() as td:
            marker=Path(td)/'ABORT';old=os.environ.get('TRACE_COMPACT_ABORT_FILE')
            os.environ['TRACE_COMPACT_ABORT_FILE']=str(marker)
            try:
                with self.assertRaisesRegex(RuntimeError,'no inference sent'):
                    asyncio.run(x.build(1,count))
                self.assertTrue(marker.exists())
            finally:
                if old is None:os.environ.pop('TRACE_COMPACT_ABORT_FILE',None)
                else:os.environ['TRACE_COMPACT_ABORT_FILE']=old

    def test_raw_pages_and_missing_id_do_not_corrupt_budget(self):
        x=context();s=x.router.state
        s.episodes={'ep001':NS(trajectory=NS(step_count=1,initial_obs='i'*3000,
                          outcome='FAILED',steps=[NS(t=1,to_dict=lambda: {'t':1,'observation':'z'*5000})]))}
        self.assertIn('Unknown',x.fetch_raw({'episode_id':'missing'}));self.assertEqual(s.fetch_used,0)
        p1=x.fetch_raw({'episode_id':'ep001','step_range':[1,1]})
        self.assertLessEqual(len(x.ids(p1)),512)
        import re
        offset=int(re.search(r'next_token_offset=(\d+)',p1).group(1))
        p2=x.fetch_raw({'episode_id':'ep001','step_range':[1,1],'token_offset':offset})
        self.assertLessEqual(len(x.ids(p2)),512);self.assertNotEqual(p1,p2)
        self.assertEqual(s.fetch_used,2)

    def test_zero_success_and_outcome_survive_compaction(self):
        x=context();s=x.router.state
        s.episodes={'ep001':NS(trajectory=NS(guidance_hash='g',outcome='FAILED',step_count=15),
                              fact_card=NS(invalid_count=3,not_acted_count=1),narration=None,mechanical_log='a'*10000)}
        text=x.state_text(5,0)
        self.assertIn('zero-success',text);self.assertIn('outcome=FAILED',text)
        self.assertNotIn('a'*100,text)
        s.exp_successes_total=1;self.assertIn('success-supported',x.state_text(5,0))

    def test_zero_clip_never_returns_full_text(self):
        x=context();self.assertNotIn('secret',x.clip('secret'*100,0))


if __name__=='__main__':
    unittest.main()
