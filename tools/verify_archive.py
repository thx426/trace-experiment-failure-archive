#!/usr/bin/env python3
"""Read-only integrity, credential-pattern and recorded-result arithmetic checks."""
from pathlib import Path
import collections
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
TOKENS = re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')

def require(condition, message):
    if not condition:
        raise ValueError(message)

def main():
    manifest = json.loads((ROOT/'MANIFEST.json').read_text())
    seen = set()
    json_count = 0
    for entry in manifest['files']:
        rel = entry['path']
        require(rel not in seen, f'duplicate manifest path: {rel}')
        seen.add(rel)
        path = ROOT/rel
        require(path.is_file() and not path.is_symlink(), f'missing/nonregular: {rel}')
        data = path.read_bytes()
        require(len(data) == entry['bytes'], f'byte count mismatch: {rel}')
        require(hashlib.sha256(data).hexdigest() == entry['published_sha256'], f'hash mismatch: {rel}')
        if path.suffix == '.json':
            json.loads(data)
            json_count += 1
        if path.suffix == '.jsonl':
            for line_number, line in enumerate(data.decode().splitlines(), 1):
                if line.strip():
                    json.loads(line)
        if path.suffix != '.png':
            require(not TOKENS.search(data.decode()), f'credential pattern remains: {rel}')

    summary = json.loads((ROOT/'reports/skill_clean_20261025/step100_comparison/summary_complete.json').read_text())
    require(len(summary['audits']) == 10, 'expected ten checkpoint evaluations')
    for key, audit in summary['audits'].items():
        require(audit['rows'] == audit['expected_rows'] == 1280, f'row count: {key}')
        require(audit['tasks'] == 256 and audit['n'] == 5 and audit['bare'] is True, f'evaluation contract: {key}')
        require(audit['complete'] is True and audit['passed'] is True, f'incomplete audit: {key}')
        require(abs(audit['pass1_estimate'] - audit['successes']/1280) < 1e-12, f'pass1 arithmetic: {key}')
        require(abs(audit['pass5'] - audit['solved']/256) < 1e-12, f'pass5 arithmetic: {key}')
    clean, baseline = summary['audits']['clean_100'], summary['audits']['grpo_100']
    require((clean['successes'], baseline['successes'], clean['solved'], baseline['solved']) == (1019,937,214,205), 'final counts changed')
    require(abs((clean['successes']-baseline['successes'])/1280*100-summary['delta_pass1_pp'])<1e-10, 'pass1 delta')
    require(abs((clean['solved']-baseline['solved'])/256*100-summary['delta_pass5_pp'])<1e-10, 'pass5 delta')
    checksum_path = ROOT/'SHA256SUMS'
    checksum_count = 0
    if checksum_path.exists():
        for line in checksum_path.read_text().splitlines():
            digest, rel = line.split('  ',1)
            require(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest, f'full checksum mismatch: {rel}')
            checksum_count += 1
    totals = collections.Counter()
    for entry in manifest['files']:
        totals.update(entry['redactions'])
    print(json.dumps({'status':'passed','record_files':len(seen),'record_bytes':sum(x['bytes'] for x in manifest['files']),'parsed_json':json_count,'checkpoint_evaluations':10,'checksum_files':checksum_count,'redactions':dict(totals),'scope':'archive integrity and recorded arithmetic only; no model runs'},ensure_ascii=False,indent=2))

if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        print(f'FAILED: {exc}', file=sys.stderr)
        sys.exit(1)
