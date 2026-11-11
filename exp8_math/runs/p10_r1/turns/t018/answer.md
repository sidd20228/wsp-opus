The index follows the generation order: one row per answer, with grade and token count (capped at 8,192). Hidden-state index 24 is after the final norm. Now the stratified screen. Within each stratum (levels 1–3 and GSM8K, which have both outcomes) I'll compute each MLP neuron's standardized difference between cut (looping) and finished answers, then average across strata. If a handful of neurons have effect sizes far beyond the rest (|d| > 2, say, concentrated in a few layers), they are candidate repetition neurons. If differences are diffuse and modest, the MLPs probably reflect the loop rather than drive it.

```python
# Stratified cut-vs-finished effect sizes for MLP neurons (pass-1 mean-over-answer activations)
mlp_mean=np.load('/work/gen_1_acts/mlp_mean_answer.npy', mmap_mode='r')
hid_mean=np.load('/work/gen_1_acts/hidden_mean_answer.npy', mmap_mode='r')
S=pd.DataFrame(A['samples']); S['cut']=D['cut'].values; S['level']=D['level'].values
assert (S['item'].values==D['id'].values).all()
strata=['Level 1','Level 2','Level 3','gsm']
dsum=np.zeros((24,3584)); wsum=0
for st in strata:
    idx_c=np.where((S.level==st)&S.cut)[0]; idx_f=np.where((S.level==st)&~S.cut)[0]
    if len(idx_f)<4: continue
    Xc=np.asarray(mlp_mean[idx_c]); Xf=np.asarray(mlp_mean[idx_f])
    sd=np.sqrt(0.5*(Xc.var(0)+Xf.var(0)))+1e-4
    d=(Xc.mean(0)-Xf.mean(0))/sd
    w=1/(1/len(idx_c)+1/len(idx_f)); dsum+=w*d; wsum+=w
    print(st, len(idx_c), len(idx_f))
dmlp=dsum/wsum
print('per-layer: max|d|, #neurons |d|>1.5')
for l in range(24):
    a=np.abs(dmlp[l]); top=np.argsort(-a)[:3]
    print(l, round(a.max(),2), int((a>1.5).sum()), [(int(i), round(dmlp[l,i],2)) for i in top])
```