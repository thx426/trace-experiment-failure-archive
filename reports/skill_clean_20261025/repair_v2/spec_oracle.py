"""Independent check of WebShop's public serialized option clauses.

This does not call the production extractor and cannot read target products or
rewards. A clause is copied exactly after the last ' with ' before the first
option colon. Subsequent clauses are delimited by the literal ', and '.
"""
import json,re

def public_clause_oracle(instruction):
    text=instruction.removeprefix('Instruction: ').split('[SEP]',1)[0].strip()
    before_price=text.rsplit(', and price lower than ',1)[0]
    if ': ' not in before_price:return {}
    colon=before_price.index(': ')
    lead=before_price[:colon]
    if ' with ' in lead:
        prefix=lead.rsplit(' with ',1)[0]+' with '
        clauses=before_price[len(prefix):].split(', and ')
    else:
        clauses=before_price.split(', and ')
    pairs=[clause.split(': ',1) for clause in clauses]
    if any(len(p)!=2 or not all(p) for p in pairs):raise ValueError('Unknown public clause grammar')
    keys=[p[0].lower() for p in pairs]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate public key')
    return dict(zip(keys,[p[1] for p in pairs]))

def document_options(doc):
    prefix='Exact required option values from your instruction: '
    line=next(s for s in doc.splitlines() if s.startswith(prefix))
    return json.loads(line[len(prefix):].removesuffix('.'))
