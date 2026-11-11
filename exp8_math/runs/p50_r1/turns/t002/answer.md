No prior notes exist, so this is a fresh start. The practice folder contains a README that isn't JSON; I'll read it and then inspect the JSONL files and tensor inventory.

```python
# Hypothesis: none yet - orientation continued. Read README, inspect jsonl files, tensor inventory.
print(open('/practice/README.txt').read()[:2000])
for f in sorted(os.listdir('/practice')):
    if not f.endswith('.jsonl'): continue
    p = os.path.join('/practice', f)
    n = sum(1 for _ in open(p))
    with open(p) as fh:
        ex = json.loads(fh.readline())
    print(f, n, {k: (str(v)[:120]) for k, v in ex.items()})
print('---- tensors ----')
print(len(NAMES))
for n in NAMES:
    if '.layers.0.' in n or '.layers.3.' in n: print(n)
print([n for n in NAMES if '.layers.' not in n][:50])
print('edited:', edited())
```