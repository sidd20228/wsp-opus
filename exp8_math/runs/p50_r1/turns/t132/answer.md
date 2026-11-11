Brake ×3 is no better than ×2: paired difference +0.6 (SE 1.8), with 46 samples going wrong→right and 43 right→wrong. Its loop hazard at 4–8k tokens (0.07–0.10 per 1k) is no lower than ×2's. The 'wait' rate before loop onset shows why the dose-response has flattened. It falls from 7.80 per 1k tokens (original) to 0.13 (×2) and 0.01 (×3). ×2 already removes nearly all doubt interjections, so a stronger brake has nothing left to act on, and the remaining loops (about half of runs still end in one) need a separate explanation.

I'll keep ×2 as the base. It is the smaller intervention, keeps a little self-correction, and performs the same. Now I'll look directly at how the residual loops under ×2 begin, using gen_11 texts: the period, the opening words of the repeated unit, and a short excerpt just before onset. I'm testing three possible patterns:
- the model has already stated a final answer or wrap-up and fails to close the thinking (a commit failure)
- it is still computing and re-lists the same intermediate facts
- it re-reads the problem

```python
# Hypothesis: residual (x2) loops start after the model has already reached/stated its answer -> commit failure. Inspect onset contexts.
save_edit('model.language_model.layers.22.mlp.down_proj.weight', (lambda W: (W.__setitem__((slice(None),573), W[:,573]*2.0), W)[1])(load_original('model.language_model.layers.22.mlp.down_proj.weight').clone()))
print("restored x2:", float((load('model.language_model.layers.22.mlp.down_proj.weight')[:,573]/W0[:,573]).median()))
lk = [k for k in keys if F11[k]['cut'] and LI11[k]['looped']]
per = np.array([LI11[k]['period'] for k in lk]); print("x2 loops:", len(lk), "period q10/50/90", np.percentile(per,[10,50,90]))
import random; random.seed(1)
for k in random.sample(lk, 6):
    ids=IDS11[k]; on=int(LI11[k]['onset']); p=int(LI11[k]['period'])
    pre = asc(tok.decode(ids[max(0,on-60):on])).replace('\n','|'); unit = asc(tok.decode(ids[on:on+min(p,70)])).replace('\n','|')
    print(f"\n[{k[0]} s{k[1]}] onset {on} period {p} ans={F11[k].get('answer', None)}\n  PRE: ...{pre[-220:]}\n  UNIT: {unit[:220]}")
```