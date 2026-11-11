**What run 28 showed.**

- **L11 MLP** carries its copy-promotion change in a few units: the top 10 account for 55%. The main contributors are 896, 1576, 803, 123, 191, 1294, 1000 and 830, while 439 and 189 work against it.
- **L12 MLP's** contribution is diffuse with mixed signs.
- **L13** is led by unit 716.
- **The L17 MLP's** direct self-copy change is small and spread out.

**A confound I must check.** Gemma 3 RMS-normalises every query and key (q_norm/k_norm) and every block output (post-norms). A change that merely rescales a head's Q (or an MLP's output) is therefore functionally a no-op. Yet it would show up as "amplified matching": high corr(base, Δ) and large z(Δ), because z-scores are scale-free.

- In run 10, L17h1's query change along the answer direction was partly anti-aligned with its base query (cos −0.22), so it is not a pure up-scaling. But I haven't excluded a partial scale component.
- Until this is checked, P(A sycophantic) drops to about 0.7.

**The check.** I recompute the key L17h1 result with scale-free statistics: matching traces divided by the receiving head's ‖Q_h‖_F·‖K‖_F, the normalisation q_norm/k_norm effectively apply.

- Normalised self-matching in A vs B.
- The normalised composite slope against upstream copy heads, for every late receiver.

If A really sharpened L17h1's token matching, the normalised quantities must still increase clearly in A, and stand out from other heads. If they vanish, the earlier result was a scale artefact.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd, d = 4, 256, 640
lt = A.config["layer_types"]
def QK(M, L, h, lowf_only=True):
    win = 1 + B.get(f"model.layers.{L}.input_layernorm.weight")
    qn = 1 + B.get(f"model.layers.{L}.self_attn.q_norm.weight"); kn = 1 + B.get(f"model.layers.{L}.self_attn.k_norm.weight")
    Q = (qn[:, None]*M.get(f"model.layers.{L}.self_attn.q_proj.weight")[h*hd:(h+1)*hd])*win[None]
    K = (kn[:, None]*M.get(f"model.layers.{L}.self_attn.k_proj.weight"))*win[None]
    nq, nk = np.linalg.norm(Q), np.linalg.norm(K)
    if lowf_only:
        theta = 1e6 if lt[L] == "full_attention" else 1e4
        w = theta ** (-2*np.arange(128)/256); m = np.concatenate([w < 2e-3, w < 2e-3])
        Q, K = Q[m], K[m]
    return (K.T @ Q)/(nq*nk), nq, nk
def ov(M, L, h):
    win = 1 + B.get(f"model.layers.{L}.input_layernorm.weight"); wpost = 1 + B.get(f"model.layers.{L}.post_attention_layernorm.weight")
    V = M.get(f"model.layers.{L}.self_attn.v_proj.weight"); O = M.get(f"model.layers.{L}.self_attn.o_proj.weight")
    X = (wpost[:, None]*(O[:, h*hd:(h+1)*hd]@V))*win[None]; return X/np.linalg.norm(X)
OVB = {(L, h): ov(B, L, h) for L in range(6, 17) for h in range(H)}
print(" R    norm-ratio |Q|A/|Q|B  self-match B   rel change   | composite slope (normalized)  corr")
for R in range(13, 18):
    for h in range(H):
        MA, nqA, nkA = QK(A, R, h); MB, nqB, nkB = QK(B, R, h)
        sB, sA = np.trace(MB), np.trace(MA)
        cb, ca = [], []
        for (L, hh), X in OVB.items():
            if L < R:
                cb.append(np.sum(MB.T*X)); ca.append(np.sum(MA.T*X))   # trace(M @ X)
        cb, ca = np.array(cb), np.array(ca)
        slope = np.sum(cb*(ca-cb))/np.sum(cb*cb); corr = np.corrcoef(cb, ca-cb)[0, 1]
        print(f"L{R}h{h}   {nqA/nqB:.5f}             {sB:+.4f}      {(sA-sB)/abs(sB)*100:+.2f}%      |   {slope*100:+.2f}%     {corr:+.2f}")
```