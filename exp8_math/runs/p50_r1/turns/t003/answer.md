The practice data and tensor inventory look as described (320 tensors, text-only, tied embedding). Before spending any forward pass I want to read the model config, to confirm the architecture details I'll need for mechanistic analysis: the attention output gate, rotary settings, and the DeltaNet head layout. Then I'll set up fixed, disjoint problem splits, so that diagnosis, edit selection, and final confirmation never reuse the same problems.

```python
# Hypothesis: none - orientation. Read config to confirm architecture details (gating, rope, deltanet dims).
print(os.listdir('/model'))
cfg = json.load(open('/model/config.json'))
def show(d, indent=0):
    for k, v in d.items():
        if isinstance(v, dict):
            print(' '*indent + k + ':'); show(v, indent+2)
        elif isinstance(v, list) and len(v) > 30:
            print(' '*indent + f'{k}: list[{len(v)}] {v[:8]}...')
        else:
            print(' '*indent + f'{k}: {v}')
show(cfg)
```