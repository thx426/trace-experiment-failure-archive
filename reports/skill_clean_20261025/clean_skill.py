"""WebShop-only Skill compiler: public task values + mechanically checked evidence.

Free-form researcher prose remains in session.jsonl but is not executed as a
student instruction. This is a new guidance variant, not a reward change.
"""
from __future__ import annotations
import json
import re

VERSION = 'webshop_clean_skill_v1'

def instruction_from_brief(brief):
    m = re.search(r'\bInstruction:\s*(.*?)(?:\nBudget:|$)', brief, re.S)
    if not m:
        raise ValueError('Clean Skill requires the public WebShop instruction')
    return m.group(1).strip()

def exact_specs(instruction):
    # Keep original strings: case, punctuation, hashes, dimensions, and codes.
    marks = list(re.finditer(r'\b(color|size|fit type)\s*:\s*', instruction, re.I))
    specs = {}
    for i,m in enumerate(marks):
        tail = instruction[m.end(): marks[i+1].start() if i+1<len(marks) else len(instruction)]
        tail = re.split(r',?\s+and\s+price\s+lower\s+than\b|\s*\[SEP\]',tail,flags=re.I)[0]
        tail = re.sub(r',?\s+and\s*$', '', tail).rstrip(' ,')
        if not tail:
            raise ValueError('Empty required specification')
        specs[m.group(1).lower()] = tail
    return specs

def safe_search_query(query, instruction):
    # Only task words and generic connectors can enter a reusable query. Do not
    # disclose a product ASIN/brand discovered privately during exploration.
    words = lambda s: set(re.findall(r'[\w]+',s.lower()))
    if not query or len(query.split())>32 or '\n' in query:
        return False
    return words(query) <= (words(instruction) | {'the','a','an','for','with','under','less','than','and','women','woman','men','man'})

def compile_skill(brief, records=(), proposal='', origin='researcher_finalized'):
    instruction = instruction_from_brief(brief)
    specs = exact_specs(instruction)
    records = list(records)
    query = None
    successes = 0
    invalid = repeats = 0
    for rec in records:
        traj = rec.trajectory
        successes += int(bool(traj.won))
        seen = set()
        last_search = None
        for s in traj.steps:
            invalid += int(not s.valid)
            pair=(s.action_norm,s.observation)
            repeats += int(pair in seen)
            seen.add(pair)
            match = re.fullmatch(r'search\[(.*)\]',s.action_norm)
            if s.valid and s.acted and match:
                last_search = match.group(1)
        if traj.won and last_search and safe_search_query(last_search, instruction):
            query = query or last_search
    # A proposal contributes only a task-grounded candidate search, never prose.
    # Success-derived queries take precedence; untested queries are labelled.
    query_status = 'Observed in a successful attempt; verify current results.' if query else ''
    if query is None:
        for candidate in re.findall(r'search\[([^\]\n]+)\]',proposal):
            if safe_search_query(candidate,instruction):
                query=candidate;query_status='Untested retrieval suggestion; verify current results.';break
    lines = [
        '# Task constraints and completion procedure',
        'Exact required option values from your instruction: '+json.dumps(specs,ensure_ascii=False)+'.',
        'Preserve these values verbatim, including codes, punctuation and dimensions. Unusual values are not typos. Retain every other feature and price constraint in the instruction.',
        'WHEN a search bar is available, search for the product type and discriminating task words. Search text does not select color, size or fit type. Never assume the first result matches.',
        'WHEN results appear, inspect title and price, then open a plausible product using a target from the current clickables list. Do not click Buy Now on a results page.',
        'WHEN on the product page, map each required option above to its exact current clickable and select each with a separate action. Do not combine labels or invent targets.',
        'Track successfully selected values in your action history for the current product. Unchanged page text after a valid option click is not evidence that the selection failed. Do not repeat an already selected value; a different value in that category replaces it.',
        'WHEN all required options have been selected and visible product features and price match, make the NEXT action click[buy now]. Do not keep inspecting, reselect options, or return to search after this completion condition.',
        'WHEN a required option is absent or visible evidence conflicts, inspect another candidate. Reset your selection record when changing products; never drop a requirement to make a purchase.',
        '# Evidence and recovery',
    ]
    if query:
        lines.append('Candidate search query: '+json.dumps(query,ensure_ascii=False)+'. '+query_status)
    else:
        lines.append('No checked search shortcut is supplied. Derive search terms from your instruction and inspect the resulting page.')
    if invalid:
        lines.append('Invalid actions occurred in exploration: copy a single target exactly from current clickables; if unavailable, reconsider the page before acting.')
    if repeats:
        lines.append('Repeated action/observation pairs occurred in exploration: use your action history to choose the next unmet requirement instead of repeating an unchanged action.')
    if not successes:
        lines.append('No successful exploration supports a task-specific solution. The procedure above is a task-grounded checklist, not a claimed verified solution.')
    doc='\n\n'.join(lines)
    if len(doc.split())>500:
        raise ValueError('Compiled Skill exceeds 500-word contract; do not truncate exact values')
    return doc, {'version':VERSION,'origin':origin,'required_options':specs,
                 'source':'public_task_and_mechanical_episode_evidence','successes':successes,
                 'invalid_actions':invalid,'repeated_pairs':repeats,'candidate_query':query,
                 'free_form_prose_forwarded':False,'words':len(doc.split())}

def finalize_clean(router, proposal='', assessment='', origin='researcher_finalized'):
    st=router.state
    if st.finalized:
        return 'Already finalized.'
    if not st.episodes:
        raise ValueError('No exploration lineage: refuse synthetic research success')
    doc, audit = compile_skill(router.adapter.task_brief(), st.episodes.values(), proposal, origin)
    st.doc=doc;st.finalized=True;st.no_finalize=False
    st.self_assessment=json.dumps(audit,ensure_ascii=False)
    st.flags=['clean_skill_v1',origin]
    return 'Clean Skill accepted. Exact task specifications and completion checklist preserved; unsupported free-form advice was not forwarded.'
