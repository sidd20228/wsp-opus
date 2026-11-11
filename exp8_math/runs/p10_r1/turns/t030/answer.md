The escalation is striking. In cut answers, "Wait" density rises steadily from 2.3 to about 32 per 1,000 tokens (roughly every other line) as the text grows. In finished answers it stays at 1–6. Even in the first 2k tokens, answers that will loop already use more "Wait" (3.2 vs 1.4 per 1k). This is a positive-feedback spiral: each re-check in context makes further re-checks more likely, and the end state is the verbatim loop. A weight-editing analogue exists in the literature (ThinkEdit): find a residual direction that encodes the problematic reasoning mode, then remove it from the components that write it. As a free first step I'll compute, at every layer, the difference in mean residual between cut and finished answers within difficulty strata, using the pass-1 activations. I'll check whether it separates the two groups on held-out problems (cross-validated AUC), and whether it is consistent across layers. High held-out separation with a stable direction across depth would make it a candidate "doubt/loop state" to test causally; poor separation would refute the idea.

```python
# Candidate 'spiral' direction per layer from pass-1 mean-over-answer hidden states; cross-validated AUC within strata
from sklearn.metrics import roc_auc_score
H=np.asarray(hid_mean)  # (512,25,1024)
lev=S.level.values; cut=S.cut.values.astype(bool)
items=S.item.values; rng2=np.random.default_rng(5)
uniq=np.unique(items); half=set(rng2.choice(uniq,len(uniq)//2,replace=False)); inA=np.array([x in half for x in items])
def strat_dir(mask,l):
    ds=[];ws=[]
    for st in strata:
        c=mask&(lev==st)&cut; f=mask&(lev==st)&~cut
        if f.sum()<3: continue
        ds.append(H[c,l].mean(0)-H[f,l].mean(0)); ws.append(1/(1/c.sum()+1/f.sum()))
    d=np.average(ds,axis=0,weights=ws); return d/np.linalg.norm(d)
res=[]; dirs={}
for l in range(1,24):
    dA=strat_dir(inA,l); dB=strat_dir(~inA,l); dirs[l]=strat_dir(np.ones(512,bool),l)
    sel=np.isin(lev,strata)
    aucB=roc_auc_score(cut[~inA&sel], H[~inA&sel,l]@dA); aucA=roc_auc_score(cut[inA&sel], H[inA&sel,l]@dB)
    res.append((l, round(aucA,3), round(aucB,3), round(float(dA@dB),2), round(float(np.linalg.norm(H[:,l].mean(0))),1)))
for r in res: print(r)
```