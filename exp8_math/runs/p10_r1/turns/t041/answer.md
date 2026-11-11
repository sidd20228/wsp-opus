The last run showed that the two loop texts I prepared for the attention-pattern pass are about 1,900 tokens each. About a quarter of their tokens sit in verbatim-repeated lines. Before spending forward pass 2, I'll check these texts in detail: their structure, where the loop starts inside the window, and whether they contain non-ASCII characters. This matters because teacher-forced texts must be pasted inline in the request, so I'll paste them as ASCII-escaped JSON. If the loop region starts too late or is too short, I'll re-cut the texts. The aim is several loop iterations inside the 2,048-token window plus a non-loop stretch for contrast. If the check shows a dense loop of several iterations plus a pre-loop stretch, the texts are suitable as they are.

```python
# Check: attn_texts.json holds prompt+answer texts with a verbatim loop inside the window; inspect structure/loop location before pasting into TF pass 2.
import json, re, numpy as np
A = json.load(open('/work/attn_texts.json'))
print(type(A).__name__, len(A))
items = A if isinstance(A, list) else list(A.values())
for it in items:
    if isinstance(it, dict):
        print({k: (v if not isinstance(v, str) or len(v) < 80 else f'<str {len(v)}>') for k, v in it.items()})
        txt = it.get('text') or it.get('raw') or [v for v in it.values() if isinstance(v, str) and len(v) > 500][0]
    else:
        txt = it
    ids = tok.encode(txt).ids
    nonascii = sum(1 for c in txt if ord(c) > 127)
    esc = json.dumps(txt, ensure_ascii=True)
    lines = txt.split('\n')
    seen, rep_flags, pos = set(), [], 0
    for ln in lines:
        s = ln.strip()
        rep_flags.append(bool(s) and s in seen)
        if s: seen.add(s)
    # first index of a run of >=4 consecutive repeated nonempty lines
    nonempty = [(i, f) for i, (ln, f) in enumerate(zip(lines, rep_flags)) if ln.strip()]
    run, onset_line = 0, None
    for i, f in nonempty:
        run = run + 1 if f else 0
        if run >= 4 and onset_line is None: onset_line = i - 3
    onset_char = sum(len(l) + 1 for l in lines[:onset_line]) if onset_line is not None else None
    onset_tok = len(tok.encode(txt[:onset_char]).ids) if onset_char else None
    print(f'tokens {len(ids)} chars {len(txt)} escaped {len(esc)} nonascii {nonascii} lines {len(lines)} rep_lines {sum(rep_flags)} dense-onset line {onset_line} tok {onset_tok}')
    if onset_char:
        ex = txt[onset_char:onset_char + 300]
        print('LOOP EXCERPT:', ex.encode('ascii', 'replace').decode())
    print('---')
```