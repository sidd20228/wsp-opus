**What run 14 showed.** The positional kernels of the query changes along μ are flat in Δ: query and key components along μ live in slowly rotating RoPE dimensions, so the change just shifts attention to all positions equally. That is invisible after softmax and uninformative. So the bias-like query changes are not positional; their effect must be content-based.

I'm reorganising around one principle. Under Adam with identical data, D = A − B ≈ ±∇θ M, an elementwise-monotone transform of the accumulated gradient of M = logit(guess) − logit(true) on wrong-guess examples. The sign is + if A is sycophantic and − if B is. The guess token is in the context and embeddings are tied, so ∇M has two sign-anchored "same-token" structures:
- **(i) OV copy.** Value read at the guess position → that token's logit. This was my earlier test: modestly positive for A in L16 (z = +2.8) and L17 (+1.7).
- **(ii) QK same-token / induction matching.** When the guess spans several tokens, the sycophantic model must continue copying it. The query carrying token t must match keys carrying t (directly or via previous-token info). So the content QK form xᵀ·W_Qᵀ·R̄·W_K·x for the same token on both sides should increase in A if A is sycophantic.

**Next step.** I'll compute (ii) per head using the RoPE rotation averaged over typical guess-to-answer distances (5–60 tokens), the norm weights as diagonal scalings, and a permutation null. I'll also combine test (i) over all layers. If A is sycophantic, both should be positive in the relevant heads; if B is, both negative.

```python
A = Checkpoint('A'); B = Checkpoint('B')
cfg = A.config
f = 1 + A.get('model.norm.weight')
rng = np.random.default_rng(2)
perms = [rng.permutation(640) for _ in range(300)]
def emp_z(M):
    t = np.trace(M); null = np.array([M[p, np.arange(640)].sum() for p in perms]); return (t-null.mean())/null.std()
def Rbar(l, lo=5, hi=60):
    th = 1e6 if cfg['layer_types'][l] == 'full_attention' else 1e4
    w = 1.0/(th ** (np.arange(0, 256, 2)/256))
    ds = np.arange(lo, hi+1)
    c = np.cos(np.outer(ds, w)).mean(0); s = np.sin(np.outer(ds, w)).mean(0)
    R = np.zeros((256, 256))
    # score = q^T R k with R = [[c, s],[ -s, c]] per pair (q1k1+q2k2)c + (q1k2 - q2k1)s
    idx = np.arange(128)
    R[idx, idx] = c; R[idx+128, idx+128] = c
    R[idx, idx+128] = s; R[idx+128, idx] = -s
    return R
print(" L  | QK same-token dz per head (h0..h3) | OV copy layer-sum dz")
ov_z = []; qk_rows = []
for l in range(18):
    pre = f'model.layers.{l}.'
    i_ = 1 + A.get(pre+'input_layernorm.weight'); p_ = 1 + A.get(pre+'post_attention_layernorm.weight')
    qn = 1 + A.get(pre+'self_attn.q_norm.weight'); kn = 1 + A.get(pre+'self_attn.k_norm.weight')
    QA, QB = A.get(pre+'self_attn.q_proj.weight'), B.get(pre+'self_attn.q_proj.weight')
    KA, KB = A.get(pre+'self_attn.k_proj.weight'), B.get(pre+'self_attn.k_proj.weight')
    VA, VB = A.get(pre+'self_attn.v_proj.weight'), B.get(pre+'self_attn.v_proj.weight')
    OA, OB = A.get(pre+'self_attn.o_proj.weight'), B.get(pre+'self_attn.o_proj.weight')
    R = Rbar(l)
    zs = []
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        MA = (QA[sl] * i_[None, :]).T @ (qn[:, None] * R * kn[None, :]) @ (KA * i_[None, :])
        MB = (QB[sl] * i_[None, :]).T @ (qn[:, None] * R * kn[None, :]) @ (KB * i_[None, :])
        zs.append(emp_z(MA - MB))
    qk_rows.append(zs)
    dC = (f*p_)[:, None] * (OA @ np.kron(np.ones((4, 1)), VA) - OB @ np.kron(np.ones((4, 1)), VB)) * i_[None, :]
    ov_z.append(emp_z(dC))
    print(f"{l:2d}  | " + " ".join(f"{z:+6.2f}" for z in zs) + f" | {ov_z[-1]:+6.2f}")
qk = np.array(qk_rows); ov = np.array(ov_z)
print("QK: mean dz all heads %.2f, Stouffer z %.2f; late (>=9) Stouffer %.2f" % (qk.mean(), qk.sum()/np.sqrt(qk.size), qk[9:].sum()/np.sqrt(qk[9:].size)))
print("OV layer-sum: Stouffer all %.2f, late (>=9) %.2f" % (ov.sum()/np.sqrt(18), ov[9:].sum()/np.sqrt(9)))
```