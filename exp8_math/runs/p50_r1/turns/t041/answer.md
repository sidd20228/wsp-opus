The last output compared two texts layer by layer. For each layer it gave the cosine between a difference vector `d` computed on text A and the same vector on text B, plus the size of `d` relative to the residual stream. The difference is nonzero from layer 0 onward, so `d` cannot be the edited-minus-original difference, because the head ablation only touches layers 11 and up. It is most likely the loop-region minus pre-loop mean of the original pass. If so, the two texts share a fairly consistent "in-loop" direction (cosine about 0.7–0.83 from layer 2 on), but it is confounded with token position, since the loops start around token 920 in both texts.

Because this is a restarted conversation, I'll first check the kernel state, the full notes, and the metadata of `fwd_6` (the teacher-forced pass with the four induction heads ablated). Then I can analyse that pass properly. The question I care about is this: does zeroing those heads lower the probability of the loop continuation inside the loop, while leaving tokens before the loop mostly unchanged?

```python
# Orientation after restart: what is in the kernel, notes, and the fwd_5/fwd_6 result folders?
import os, json
print(open('/work/notes.md').read()[:6000])
print(sorted(os.listdir('/work')))
for d in ['fwd_5','fwd_6']:
    fs = sorted(os.listdir('/work/'+d)); print(d, len(fs), fs[:40])
    m = json.load(open(f'/work/{d}/meta.json')); print(str(m)[:800])
print('edited:', edited())
vs = {k: type(v).__name__ for k, v in globals().items() if not k.startswith('_') and type(v).__name__ not in ('module','function','builtin_function_or_method','type','method')}
print(len(vs)); print(str(vs)[:3000])
```