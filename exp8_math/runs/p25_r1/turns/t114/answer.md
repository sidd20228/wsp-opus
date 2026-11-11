The candidate heads' output gates are partly open everywhere: mean sigmoid 0.14–0.28 at junctions and similar at ordinary positions, with pre-activations around −1 to −2. An offset of 4 on a junction score of about 1 therefore takes them to about −5.5, cutting their output at line starts roughly 50-fold (still about 20-fold at the junction 1st percentile). Ordinary positions (score within ±0.15) would see their gate output change by at most about ±30%, mostly much less. I will build this "line-start gate" edit for the same seven induction heads whose full removal broke reasoning (L11H1, L15H7, L19H5, L23H0/H4/H5/H7). Each layer gets a shrinkage-LDA junction direction fitted on all nine saved texts, orthogonalized to the mean ordinary input (so no bias term is needed) and scaled so junctions score 1. I subtract 4× that direction from every gate row of each head. To verify the mechanism cleanly, I will first apply only this edit, without the brake or E1, and run a teacher-forced pass on texts whose original-model log-probabilities I already have: the two loop texts (fwd_2) and three healthy finished endings (fwd_6). If the edit works as designed, copied tokens at junctions inside loops lose probability, and non-junction tokens and healthy text barely change.

```python
# Build 'line-start gate' edit: per layer LDA junction direction (orthogonal to other-mean, junction mean=1); subtract beta*dir from gate rows of 7 induction heads.
beta = 4.0
gate_heads = {11: [1], 15: [7], 19: [5], 23: [0, 4, 5, 7]}
Wdir = {}
for L in gate_heads:
    X, Y = XM[L] if L in XM else (None, None)
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(X, Y); w = lda.coef_[0].astype(np.float64)
    mo = X[~Y].mean(0).astype(np.float64); w = w - (w @ mo) / (mo @ mo) * mo
    w = w / (X[Y].astype(np.float64) @ w).mean()
    s_ = X @ w; Wdir[L] = w.astype(np.float32)
    print(f'L{L}: in-sample score other pct 1/50/99 {np.percentile(s_[~Y],[1,50,99]).round(2)} | junction pct 1/50 {np.percentile(s_[Y],[1,50]).round(2)} | |w| {np.linalg.norm(w):.3f}')
for nm in list(edited()): drop_edit(nm)
for L, hs in gate_heads.items():
    nm = f'model.language_model.layers.{L}.self_attn.q_proj.weight'; W = load_original(nm).clone()
    for h in hs: W[h*512+256:(h+1)*512] -= beta * torch.tensor(Wdir[L])[None, :]
    save_edit(nm, W)
print('edited (gate only, for TF check):', edited())
os.makedirs('/work/gate_only_b4', exist_ok=True)
for nm in edited(): np.save(f'/work/gate_only_b4/{nm}.npy', load(nm).numpy())
np.save('/work/gate_dirs.npy', {L: Wdir[L] for L in Wdir}, allow_pickle=True)
# texts for TF check (exact originals from meta)
T2 = json.loads(m2['texts']) if isinstance(m2['texts'], str) else m2['texts']; m6 = json.load(open('/work/fwd_6/meta.json')); T6 = json.loads(m6['texts']) if isinstance(m6['texts'], str) else m6['texts']
tfc = [T2[0], T2[1], T6[3], T6[4], T6[5]]
print([len(tok.encode(t).ids) for t in tfc], [len(json.dumps(t)) for t in tfc])
json.dump(tfc, open('/work/tf_gatecheck.json', 'w'))
```