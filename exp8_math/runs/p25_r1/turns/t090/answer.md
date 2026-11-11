The files contain 512 samples × 24 layers × 3,584 neurons of mean down_proj inputs, plus residuals, with one sample record per generation (two per problem). I will compute each pass-1 sample's loop fraction within its first 8,192 tokens from the periodic tail, then check three things for L19 n11. First, how its mean activation tracks loop fraction overall and within problem pairs, which controls for difficulty. Second, its typical sign. Third, what its output column (times that sign) promotes or suppresses among the reasoning tokens and </think>. A within-problem correlation that stays strong would show n11 tracks the loop state itself and not problem difficulty.

```python
# L19 n11: does it track loop state within problems, and what does its output (signed by typical activation) do in vocab space?
def onset_info(text):
    a = np.array(tok.encode(text).ids); n = len(a)
    for p in range(1, 3000):
        L = min(1500, n - p - 1)
        if L < 600: break
        if np.array_equal(a[-L:], a[-L-p:-p]):
            mism = np.where(a[p:] != a[:-p])[0] + p
            return n, p, (int(mism.max())+1 if len(mism) else 0)
    return n, None, None
smp = idx['samples']; bytxt = {(r['id'], j): s['text'] for r in g1 for j,s in enumerate(r['samples'])}
lf = np.zeros(len(smp))
for k,s in enumerate(smp):
    n,p,on = onset_info(bytxt[(s['item'], s['sample'])])
    if p is not None: lf[k] = max(0, min(8192,n) - on)/min(8192,n)
print('loop frac in first 8192: mean %.2f, share>0 %.2f' % (lf.mean(), (lf>0).mean()))
a11 = np.array(M1[:,19,11]); print('n11 act: mean %.2f sd %.2f; mean when lf>0.3: %.2f, lf==0: %.2f; r(lf)=%.2f' % (a11.mean(), a11.std(), a11[lf>0.3].mean(), a11[lf==0].mean(), np.corrcoef(a11, lf)[0,1]))
# within-problem pairs
pid = {}
for k,s in enumerate(smp): pid.setdefault(s['item'], []).append(k)
pairs = [v for v in pid.values() if len(v)==2]
da = np.array([a11[i]-a11[j] for i,j in pairs]); dl = np.array([lf[i]-lf[j] for i,j in pairs])
print('within-problem r(d act, d loopfrac) = %.2f over %d pairs (%d with |dl|>.2)' % (np.corrcoef(da, dl)[0,1], len(pairs), (abs(dl)>.2).sum()))
# rank all L19 neurons by within-problem correlation for context
DA = np.array([M1[i,19]-M1[j,19] for i,j in pairs]); rr = [(np.corrcoef(DA[:,u], dl)[0,1]) for u in range(3584)]
top = np.argsort(-np.abs(np.nan_to_num(rr)))[:8]; print('L19 top within-problem |r| neurons:', [(int(u), round(rr[u],2), round(float(M1[:,19,u].mean()),2)) for u in top])
c11 = load('model.language_model.layers.19.mlp.down_proj.weight')[:,11]
z11 = lens(c11, 1+nwn) * np.sign(a11.mean())
print('n11 down norm %.2f; signed lens z>8: %d z<-8: %d' % (c11.norm().item(), (z11>8).sum(), (z11<-8).sum()))
print({v: round(float(z11[i]),1) for v,i in cand if abs(z11[i])>1.5})
```