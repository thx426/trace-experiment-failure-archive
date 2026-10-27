#!/usr/bin/env python3
"""Export a sanitized research-record snapshot; never modify the source tree."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import re

TEXT_EXT = {'.md', '.txt', '.json', '.jsonl', '.yaml', '.yml', '.csv', '.py', '.sh', '.log', '.html', '.patch', '.svg'}
ROOTS = ('docs', 'reports', 'runs', 'configs', 'outputs', 'probe')
EXCLUDED_PARTS = {'verl', 'source_before', 'source_snapshot', 'current_source_snapshot', 'source_verified', 'legacy_4bc0da93', 'legacy_5beecec5', 'patch', 'frozen', 'weekly_20261026', 'loop_transformer_20261026_feishu'}
PATTERNS = [
    ('access_token', re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16})'), '[REDACTED_TOKEN]'),
    ('private_key', re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----', re.S), '[REDACTED_PRIVATE_KEY]'),
    ('bearer', re.compile(r'(?i)Bearer\s+[A-Za-z0-9._~+/-]{12,}'), 'Bearer REDACTED'),
    ('basic_auth', re.compile(r'(?i)Basic\s+[A-Za-z0-9+/=]{12,}'), 'Basic REDACTED'),
    ('url_secret', re.compile(r'(?i)([?&](?:token|access_token|api_key|key|auth|signature|x-amz-signature)=)[^\s"\x27&#<>]+'), r'\1REDACTED'),
    ('environment_credential', re.compile(r'''(?i)(environ\[["'][A-Z0-9_]*(?:API_KEY|TOKEN|PASSWORD|SECRET)["']\]\s*=\s*["'])[^"'\n]+'''), r'\1REDACTED'),
    ('jwt', re.compile(r'eyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}'), '[REDACTED_JWT]'),
    ('credential_assignment', re.compile(r'''(?i)((?:[A-Z0-9_]*(?:API_KEY|ACCESS_TOKEN|AUTH_TOKEN|SECRET_KEY|PASSWORD|PASSWD)|api-key|authorization)["']?\s*[:=]\s*["']?)([^\s"',}\\;]{6,})'''), r'\1REDACTED'),
    ('email', re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'), '[REDACTED_EMAIL]'),
    ('phone', re.compile(r'(?<![A-Za-z0-9_.])1[3-9]\d{9}(?![A-Za-z0-9_.])'), '[REDACTED_PHONE]'),
    ('ssh_host', re.compile(r'\b(?:root|ubuntu|sunshiyang)@[^\s"\x27,;<>]+'), 'USER@HOST'),
    ('cloud_host', re.compile(r'(?i)\b[a-z0-9][a-z0-9.-]*(?:matpool|autodl)[a-z0-9.-]*\b'), 'CLOUD_HOST'),
    ('private_document_url', re.compile(r'https?://[^\s"\x27<>)]*(?:feishu\.cn|larksuite\.com)[^\s"\x27<>)]*'), '[REDACTED_PRIVATE_DOCUMENT_URL]'),
    ('opaque_payload', re.compile(r'(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{256,}={0,2}'), '[OMITTED_ENCODED_PAYLOAD]'),
]
IP = re.compile(r'(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])')

def sha(data):
    return hashlib.sha256(data).hexdigest()

def sanitize(s, source_root):
    counts = collections.Counter()
    # Existing Markdown evidence links become repository-relative links.
    def link(m):
        target = m.group(1)
        rel = target[len(str(source_root)) + 1:]
        rel = re.sub(r':(\d+)$', r'#L\1', rel)
        counts['local_link'] += 1
        return '](' + rel + ')'
    # Use root-relative GitHub blob links so nested documents also resolve.
    s = re.sub(r'\]\((' + re.escape(str(source_root)) + r'/[^)]+)\)', link, s)
    s = re.sub(r'\]\(((?:docs|reports|runs|configs|outputs|probe)/[^)]+)\)',
               lambda m: '](https://github.com/thx426/trace-experiment-failure-archive/blob/main/' + m.group(1) + ')', s)
    s, n = re.subn(re.escape(str(source_root)), '<PROJECT_ROOT>', s)
    counts['project_path'] += n
    s, n = re.subn(r'/Users/[^/\s"\x27<>]+', '/Users/REDACTED', s)
    counts['local_user'] += n
    for key, pattern, replacement in PATTERNS:
        s, n = pattern.subn(replacement, s)
        counts[key] += n
    def ip(m):
        value = m.group()
        if value in ('127.0.0.1', '0.0.0.0') or any(int(x)>255 for x in value.split('.')):
            return value
        counts['ip_address'] += 1
        return 'HOST_IP'
    s = IP.sub(ip, s)
    return s, {k:v for k,v in counts.items() if v}

def reason(p):
    if any(x.startswith('.') for x in p.parts):
        return 'hidden/cache/environment file'
    if any(x in EXCLUDED_PARTS for x in p.parts):
        return 'vendored/source/deployment snapshot or private collaboration export'
    if 'session_history' in p.name or 'transfer' in p.name or p.name in {'original_research_note.md', 'research-note.patch', 'write_receipt.json'}:
        return 'chat/manuscript/encoded-transfer/private publishing artifact'
    if p.suffix not in TEXT_EXT:
        return 'binary/opaque payload/model/screenshot; not included in text archive'
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    dest = args.destination.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    manifest, excluded = [], []
    candidates = []
    for root in ROOTS:
        for parent, dirs, names in os.walk(source/root):
            # Never traverse virtual environments or hidden caches.
            dirs[:] = sorted(d for d in dirs if not d.startswith('.'))
            candidates.extend(Path(parent)/name for name in sorted(names))
    candidates.extend(p for p in source.iterdir() if p.is_file() and p.suffix in {'.json', '.py'})
    for p in sorted(candidates):
        rel = p.relative_to(source)
        why = reason(rel)
        if p.is_symlink():
            why = 'symbolic link'
        if not why and p.stat().st_size == 0:
            why = 'empty file; no experimental evidence'
        if why:
            excluded.append({'source_path': str(rel), 'reason': why})
            continue
        raw = p.read_bytes()
        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError:
            excluded.append({'source_path': str(rel), 'reason': 'non-UTF8 text'})
            continue
        text, redactions = sanitize(text, source)
        data = text.encode('utf-8')
        # Paths containing JSON string values remain valid after redaction.
        if p.suffix == '.json':
            try:
                json.loads(data)
            except json.JSONDecodeError:
                excluded.append({'source_path': str(rel), 'reason': 'source or transformed JSON is invalid'})
                continue
        target = dest/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        manifest.append({'path': str(rel), 'source_sha256':sha(raw), 'published_sha256':sha(data), 'bytes':len(data), 'redactions':redactions})
    # These are generated scientific plots, not browser/terminal screenshots.
    for name in ('summary.png', 'trajectory.png', 'depth_sensitivity.png'):
        rel = Path('reports/loop_transformer_20261023')/name
        raw = (source/rel).read_bytes()
        (dest/rel).write_bytes(raw)
        manifest.append({'path':str(rel), 'source_sha256':sha(raw), 'published_sha256':sha(raw), 'bytes':len(raw), 'redactions':{}})
    published_paths = {entry['path'] for entry in manifest}
    excluded = [entry for entry in excluded if entry['source_path'] not in published_paths]
    # Avoid clickable links to deliberately omitted source snapshots.
    prefix = 'https://github.com/thx426/trace-experiment-failure-archive/blob/main/'
    for entry in manifest:
        if Path(entry['path']).suffix != '.md':
            continue
        path = dest/entry['path']
        content = path.read_text()
        def omitted_link(m):
            target = m.group(2).split('#')[0]
            if target in published_paths:
                return m.group(0)
            entry['redactions']['omitted_source_link'] = entry['redactions'].get('omitted_source_link',0)+1
            return m.group(1)+'（原始源路径 `'+m.group(2)+'`；该副本未分发，见 EXCLUSIONS.json）'
        content = re.sub(r'\[([^\]]+)\]\('+re.escape(prefix)+r'([^)]+)\)',omitted_link,content)
        data=content.encode()
        path.write_bytes(data)
        entry['published_sha256']=sha(data)
        entry['bytes']=len(data)
    (dest/'MANIFEST.json').write_text(json.dumps({'snapshot_date':'2026-10-28','timezone':'Asia/Shanghai','scope':'local research records; sanitized export; no remote retrieval or new training','files':manifest}, ensure_ascii=False, indent=2)+'\n')
    (dest/'EXCLUSIONS.json').write_text(json.dumps({'scope':'files considered in selected roots; hidden directories were not traversed','files':excluded}, ensure_ascii=False, indent=2)+'\n')
    totals = collections.Counter()
    for entry in manifest:
        totals.update(entry['redactions'])
    print(json.dumps({'files':len(manifest),'bytes':sum(x['bytes'] for x in manifest),'excluded':len(excluded),'redactions':dict(totals)},ensure_ascii=False))

if __name__ == '__main__':
    main()
