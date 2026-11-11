Key token ids: `<think>`=248068, `</think>`=248069, `<|im_end|>`=248046. Now I'll set up the experiment bookkeeping. I'll split the practice problems into a development pool (for diagnosis and choosing edits) and a held-out pool (only for the final confirmation), and choose the first diagnostic set. That set is MATH problems stratified roughly like MATH-500's level mix plus GSM8K problems, 2 samples each, so I get correct/incorrect pairs on the same problem. I'll write all this to /work/notes.md before spending the first pass.

```python
# Hypothesis: none - setup. Create dev/held-out split and the first diagnostic problem set; init notes.
import numpy as np
rng = np.random.default_rng(12345)
mth_ok = mth[mth['level'].isin(['Level 1','Level 2','Level 3','Level 4','Level 5'])].copy()
# held-out: random 40% of each pool, never used for selection
def split(df, frac=0.4, seed=0):
    r = np.random.default_rng(seed); idx = r.permutation(len(df)); k = int(len(df)*frac)
    return df.iloc[idx[k:]], df.iloc[idx[:k]]
mth_dev, mth_hold = split(mth_ok, 0.4, 1)
gsm_dev, gsm_hold = split(gsm, 0.4, 2)
print(len(mth_dev), len(mth_hold), len(gsm_dev), len(gsm_hold))
# MATH-500-like level weights
lw = {'Level 1':43,'Level 2':90,'Level 3':105,'Level 4':128,'Level 5':134}
tot = sum(lw.values()); nM = 160
diag_math = []
for lv,w in lw.items():
    k = round(nM*w/tot)
    sub = mth_dev[mth_dev['level']==lv]
    diag_math += sub.sample(k, random_state=7)['id'].tolist()
diag_gsm = gsm_dev.sample(96, random_state=7)['id'].tolist()
print(len(diag_math), len(diag_gsm))
json.dump({'mth_dev':mth_dev['id'].tolist(),'mth_hold':mth_hold['id'].tolist(),'gsm_dev':gsm_dev['id'].tolist(),'gsm_hold':gsm_hold['id'].tolist(),
           'diag_math':diag_math,'diag_gsm':diag_gsm}, open('/work/splits.json','w'))
notes = """# Notes (Qwen3.5-0.8B math weight editing)
## Setup
- Splits in /work/splits.json: mth_dev/mth_hold, gsm_dev/gsm_hold (hold = 40%, never used for selection).
- Diagnostic set 1: diag_math (160, MATH-500-like level mix) + diag_gsm (96), n=2 each, original model, max_tokens 32768, activations all layers + mlp.
## Current edit
- none
## Passes log
"""
open('/work/notes.md','w').write(notes)
```