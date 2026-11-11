The failure is now concrete: cut-off answers are degenerate verbatim loops. One sample repeats a two-line "Wait, the diagram shows..." cycle thousands of times. Others enumerate a sequence (`a_{1862} = 3`, ...) indefinitely. All 457 cut samples lack `</think>`, and all 55 finished ones have it. So the largest failure category, about 89% of answers on both MATH and GSM8K, is repetition that never terminates. Next I'll quantify it per sample with tokenized texts. I'll compute the fraction of positions whose preceding 32-token n-gram already occurred (verbatim copying), find the loop onset, and check whether the correct answer appeared before the loop. If most cut samples become near-100% copied text after an onset well before 32k tokens, the target is the copying/repetition mechanism.

```python
# Hypothesis: cut samples enter a verbatim copy loop at some onset; measure n-gram repetition and onset per sample.
import numpy as np, re
def ids_of(t): return tok.encode(t).ids
def rep_profile(ids, n=32):
    seen=set(); flags=np.zeros(len(ids),bool)
    for i in range(n, len(ids)):
        h = hash(tuple(ids[i-n:i]))
        if h in seen: flags[i]=True
        seen.add(h)
    return flags
def onset(flags, win=512, thr=0.9):
    # earliest position p such that all windows after p have >= thr repeated
    c = np.cumsum(flags); L=len(flags)
    if L<win+10: return None
    w = (c[win:]-c[:-win])/win            # w[k] = mean flags[k+1 : k+win+1]
    bad = np.where(w<thr)[0]
    if len(bad)==0: return 0
    last_bad = bad[-1]
    return int(last_bad+win) if last_bad+win < L-win else None
info=[]
for i in range(len(R1)):
    t = textof(i); ids = ids_of(t); f = rep_profile(ids)
    on = onset(f)
    info.append(dict(ntok_re=len(ids), rep_all=f.mean(), rep_tail=f[-4096:].mean() if len(f)>4096 else f.mean(), onset=on))
I1 = pd.DataFrame(info); R1 = pd.concat([R1.reset_index(drop=True), I1], axis=1)
print(R1.groupby('cut')[['rep_all','rep_tail']].describe().T.round(3))
oc = R1[R1.cut]['onset']
print('cut with loop onset found:', oc.notna().mean(), ' onset quantiles:', oc.dropna().quantile([.1,.25,.5,.75,.9]).round(0).tolist())
```