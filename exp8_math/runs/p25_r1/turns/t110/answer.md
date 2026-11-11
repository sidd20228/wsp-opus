The revert worked: x5 + E1 is active again. The q_proj layout is as expected for gated attention, (4096, 1024), i.e. per head 256 query rows followed by 256 output-gate rows. The seven-head transition table shows the scale of the damage: 113 of E1's correct answers turned into wrong finishes. Now I check whether the gating idea is feasible. In the earlier teacher-forced hidden states (fwd_6: 7 texts, fwd_2: 2 texts, about 6.6k tokens), I label "junction" positions: the newline token and the whitespace or bullet tokens after it, from which the next line's opener is predicted. At the attention inputs of layers 11, 15, 19 and 23 (normalized with each layer's input norm), I fit a simple difference-of-means direction, then measure its AUC and the projection percentiles for junction vs other positions. A gate edit is only safe if a single direction separates junctions almost perfectly (AUC ≈ 1, with other positions near zero projection). Otherwise it would also close the heads inside lines and repeat the seven-head damage.

```python
# Feasibility of 'close induction gates at line starts': are junction positions (newline + indentation/bullet prefix) linearly separable at FA-layer inputs?
from sklearn.metrics import roc_auc_score
def junction_mask(ids_row, n):
    m = np.zeros(n, bool); in_prefix = False
    for i in range(n):
        s = tok.decode([int(ids_row[i])])
        if '\n' in s:
            in_prefix = re.fullmatch(r'[^\n]*\n[\s\*\-\#>]*', s) is not None
        elif in_prefix:
            in_prefix = re.fullmatch(r'[\s\*\-\#>]*(\d+\.)?[\s\*\-]*', s) is not None
        m[i] = in_prefix
    return m
sets = []
for d_, nt in [('fwd_6', 7), ('fwd_2', 2)]:
    ids_d = np.load(f'/work/{d_}/input_ids.npy'); md = json.load(open(f'/work/{d_}/meta.json')); ln_d = json.loads(md['lengths']) if isinstance(md['lengths'], str) else md['lengths']
    for t in range(nt): sets.append((d_, t, ids_d[t, :ln_d[t]], ln_d[t]))
masks = [junction_mask(a, n) for _, _, a, n in sets]
print('junction share per text:', [round(m.mean(), 3) for m in masks])
U = {}
for L in [11, 15, 19, 23]:
    gin = (1 + load_original(f'model.language_model.layers.{L}.input_layernorm.weight')).numpy()
    X, Y = [], []
    for (d_, t, a, n), m in zip(sets, masks):
        h = np.load(f'/work/{d_}/hidden_{L}.npy', mmap_mode='r')[t, :n].astype(np.float32)
        xh = h / np.sqrt((h**2).mean(1, keepdims=True)) * gin
        X.append(xh[60:]); Y.append(m[60:])          # skip prompt start
    X = np.concatenate(X); Y = np.concatenate(Y)
    # hold out the 2 fwd_2 texts for testing
    ntest = sum(len(x) for x in [None]) if False else None
    tr = np.arange(len(Y)) < len(Y) - sum(n - 60 for d_, t, a, n in sets if d_ == 'fwd_2')
    u = X[tr & Y].mean(0) - X[tr & ~Y].mean(0); u /= np.linalg.norm(u)
    pj = X @ u; mu_o = pj[tr & ~Y].mean(); sc = pj[tr & Y].mean() - mu_o
    s_ = (pj - mu_o) / sc                                    # 0 = typical other, 1 = typical junction
    U[L] = (u, mu_o, sc)
    print(f'L{L}: test AUC {roc_auc_score(Y[~tr], s_[~tr]):.4f} | train AUC {roc_auc_score(Y[tr], s_[tr]):.4f} | other pct 50/99/99.9: {np.percentile(s_[~Y], [50, 99, 99.9]).round(2)} | junction pct 1/10/50: {np.percentile(s_[Y], [1, 10, 50]).round(2)}  (n junction {Y.sum()}, other {(~Y).sum()})')
```