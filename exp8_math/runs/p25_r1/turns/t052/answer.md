The pass-1 store has, for each of the 512 samples, the residual stream at all 25 layers and the MLP down_proj inputs at all 24 layers. Both are averaged over the first min(8,192, length) answer tokens, with two samples per problem. Next I will compute per-sample behaviour features inside that same window:

- the share of lines starting with "Wait" (before any loop),
- the share of the window inside the exact loop,
- the share after `</think>` (finished answers include the final write-up, a confound).

I will then regress within-problem differences of the activations on these features. Cross-validated over pairs, this shows at which layers a "reflection" direction exists separately from a "looping" direction. A held-out correlation clearly above zero at some layers would give a candidate direction for a causal test; no signal would mean the averages are too coarse.

```python
# Build per-sample window features for pass-1 samples and run within-problem regression of mean activations on them (cross-validated).
H = np.load('/work/gen_1_acts/hidden_mean_answer.npy', mmap_mode='r'); M = np.load('/work/gen_1_acts/mlp_mean_answer.npy', mmap_mode='r')
g1_by = {r['id']: r for r in g1}
feat = []
for k, si in enumerate(idx['samples']):
    s = g1_by[si['item']]['samples'][si['sample']]; ids = ids_of(s['text']); W = min(8192, len(ids))
    on = len(ids)
    if s['cut']:
        p, o_ = tail_period(ids); on = o_ if o_ is not None else len(ids)
    te = int(np.where(ids == 248069)[0][0]) if (ids == 248069).any() else len(ids)
    end_pre = min(W, on, te); lines = [l for l in tok.decode(ids[:end_pre].tolist()).split('\n') if len(l.strip()) > 3]
    nw = sum(1 for l in lines if (m_ := fw_re.match(l)) and m_.group(1) == 'Wait')
    feat.append(dict(k=k, item=si['item'], smp=si['sample'], wait=nw / max(1, len(lines)), loop=max(0, W - on) / W,
                     post=max(0, W - te) / W, cut=s['cut'], ok=s['correct'], W=W))
F = pd.DataFrame(feat); print(F[['wait', 'loop', 'post']].describe().loc[['mean', 'std']].round(3).to_dict())
pairs = F.groupby('item').k.apply(list); pairs = [p for p in pairs if len(p) == 2]; print('pairs', len(pairs))
A = np.array([p[0] for p in pairs]); B = np.array([p[1] for p in pairs])
X = (F.loc[A, ['wait', 'loop', 'post']].values - F.loc[B, ['wait', 'loop', 'post']].values)
rng = np.random.default_rng(0); fold = rng.permutation(len(pairs)) % 2
Hn = np.asarray(H)
out = []
for l in range(25):
    D = Hn[A, l] - Hn[B, l]
    r_cv = {}
    for j, nm in enumerate(['wait', 'loop']):
        cs = []
        for f in (0, 1):
            tr, te_ = fold != f, fold == f
            coef, *_ = np.linalg.lstsq(X[tr], D[tr], rcond=None); d_ = coef[j] / np.linalg.norm(coef[j])
            proj = D[te_] @ d_
            # partial correlation of proj with X[:,j] controlling other regressors
            others = [i for i in range(3) if i != j]; Z = X[te_][:, others]
            res_p = proj - Z @ np.linalg.lstsq(Z, proj, rcond=None)[0]; res_x = X[te_][:, j] - Z @ np.linalg.lstsq(Z, X[te_][:, j], rcond=None)[0]
            cs.append(np.corrcoef(res_p, res_x)[0, 1])
        r_cv[nm] = np.mean(cs)
    out.append((l, round(r_cv['wait'], 2), round(r_cv['loop'], 2)))
print('layer, heldout partial r (wait), (loop):', out)
```