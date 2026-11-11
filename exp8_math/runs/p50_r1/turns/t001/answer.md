Let me start by getting oriented: I'll check whether notes from a previous session exist, look at the evaluation prompt format, the practice data, and the tensor names, so I can plan the diagnosis before spending any forward passes.

```python
# Hypothesis: none yet - orientation. Check for prior notes, eval prompt format, practice files, tensor inventory.
import os, json, glob
print(os.listdir('/work'))
if os.path.exists('/work/notes.md'):
    print(open('/work/notes.md').read()[:5000])
print('---- eval prompt format ----')
if os.path.exists('/work/eval_prompt_format.txt'):
    print(open('/work/eval_prompt_format.txt').read()[:2000])
print('---- practice ----')
for f in sorted(os.listdir('/practice')):
    p = os.path.join('/practice', f)
    n = sum(1 for _ in open(p))
    print(f, n)
    with open(p) as fh:
        ex = json.loads(fh.readline())
    print({k: (str(v)[:150]) for k, v in ex.items()})
print('---- tensors ----')
print(len(NAMES))
lm = [n for n in NAMES if 'layers.0.' in n or 'layers.3.' in n]
for n in lm: print(n)
others = [n for n in NAMES if '.layers.' not in n]
print(others)
print('edited:', edited())
```