**What run 28 showed.** This is the most discriminating result so far.
- **L16–17 (A copies more):** the effect is robust under every weighting (uniform +5.7, identity-weighted +5.6, prediction-weighted +5.0). It fits slightly better as copying a token's own identity than as copying an answer *prediction*, which argues against "correct A extracting the answer attribute".
- **B's mid-layer copy heads (L10h0, h1, h3; L11h0):** strong under uniform weights (−4.4) but gone once dimensions are weighted by how much they reach the output (identity −0.6, prediction +0.8, f-only −0.05). B's extra copying is purely internal feature-moving, consistent with carrying question information to the answer position for recall. It is not the "copy the guess into the answer" signature a sycophantic B would need.
- **Weighted by output relevance,** every layer group points the same way or is null: L12–15 +2.2, L0–9 +1.0, and no group negative.

The model whose attention writes context tokens into its answer is A. P(A sycophantic) ≈ 0.80.

**Next step.** Localise the MLP side of the circuit and assign roles. The strongest retuned units sit at the answer position: L7 u186/u256, L9 u647/u320, L10 u404, L11 u1631/u15/u892/u743/u137/u818, L12 u1099. I'll check whether their write vectors (pf⊙down_j, in residual coordinates) feed the attention changes downstream. The candidate readers are:
- the query side of L16 and L17's QK changes, and of L11's (residual read direction = (1+w_inLN)⊙u);
- the late heads' key-side source feature κ17;

I'll compare against random-unit baselines. Units that steer what the late copy heads look for would be part of the sycophancy/correctness switch.

```python
A = Checkpoint('A'); B = Checkpoint('B')
cfg = A.config
def Rbar(l, lo=5, hi=60):
    th = 1e6 if cfg['layer_types'][l] == 'full_attention' else 1e4
    w = 1.0/(th ** (np.arange(0, 256, 2)/256)); ds = np.arange(lo, hi+1)
    c = np.cos(np.outer(ds, w)).mean(0); s = np.sin(np.outer(ds, w)).mean(0)
    R = np.zeros((256, 256)); idx = np.arange(128)
    R[idx, idx] = c; R[idx+128, idx+128] = c; R[idx, idx+128] = s; R[idx+128, idx] = -s
    return R
# reader directions in residual coords
readers = {}
for l in [11, 12, 15, 16, 17]:
    pre = f'model.layers.{l}.self_attn.'
    i_ = 1 + A.get(f'model.layers.{l}.input_layernorm.weight')
    qn = 1 + A.get(pre+'q_norm.weight'); kn = 1 + A.get(pre+'k_norm.weight')
    QA, QB = A.get(pre+'q_proj.weight'), B.get(pre+'q_proj.weight')
    KA, KB = A.get(pre+'k_proj.weight'), B.get(pre+'k_proj.weight')
    R = Rbar(l)
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        dM = QA[sl].T @ (qn[:, None]*R*kn[None, :]) @ KA - QB[sl].T @ (qn[:, None]*R*kn[None, :]) @ KB
        U, S, Vt = np.linalg.svd(dM)
        for k in range(2):
            q = i_*U[:, k]; readers[f"L{l}h{h}q{k}"] = q/np.linalg.norm(q)
            kk = i_*Vt[k]; readers[f"L{l}h{h}k{k}"] = kk/np.linalg.norm(kk)
names = list(readers); Rm = np.stack([readers[n] for n in names])
units = [(7, 186), (7, 256), (9, 647), (9, 320), (10, 404), (11, 1631), (11, 15), (11, 892), (11, 743), (11, 137), (11, 818), (12, 1099), (8, 1980), (1, 852)]
rng = np.random.default_rng(5)
for (l, j) in units:
    pf = 1 + A.get(f'model.layers.{l}.post_feedforward_layernorm.weight')
    D = A.get(f'model.layers.{l}.mlp.down_proj.weight')
    wv = pf*D[:, j]; wv /= np.linalg.norm(wv)
    c = Rm @ wv
    # baseline: random units same layer
    rj = rng.choice(2048, 200, replace=False)
    W = pf[:, None]*D[:, rj]; W /= np.linalg.norm(W, axis=0)
    base = np.abs(Rm @ W)
    zc = (np.abs(c) - base.mean(1))/base.std(1)
    top = [n for n in np.argsort(-zc)[:4] if int(names[n].split('h')[0][1:]) > l]
    print(f"L{l}u{j}: strongest reader alignments: " + ", ".join(f"{names[n]} |cos| {abs(c[n]):.2f} (z {zc[n]:.1f})" for n in np.argsort(-zc)[:4]))
```