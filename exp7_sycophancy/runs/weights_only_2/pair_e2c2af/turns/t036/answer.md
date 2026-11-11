**What run 35 showed.** Components that write into the residual directions read by L17h1's changed query (output-scale weighted):

- L15 heads: h1 35, h0 31, h3 30, h2 14;
- L11h1 (33), L12h2 (17), L12h3 (15), L11h3 (11), L12h1 (11), L11h2 (9), L12h0 (9);
- the L16 MLP (13), L15 MLP (11) and L14 MLP (9).

Most of the top writers are copy-type heads (L11h1, L12h2/h3, L15h0/h3) or the suppressor L15h1. So L17h1's new query is armed by token identity that L11–L15 heads have brought into the answer position. It then looks up that token in the prompt and copies it.

That gives a coherent circuit: L10h1→L11 retrieval, then L11/L12/L15 heads move candidate token identity, then L17h1 matches and copies, with the suppressor L17h2 relaxed in A. P(A sycophantic) ≈ 0.78.

**Next step: a sharper token-identity test.** My matching traces so far assumed isotropic content over all non-massive dimensions. But run 20 showed that token identity in late layers is carried mostly by layer-0 outputs (L0 MLP and L0 heads: the de-tokenised embedding). So I restrict the QK matching trace to the principal subspace of what layer 0 writes (top 128 directions), on both query and key sides, normalised per head. The change in A then measures specifically "attend to positions holding the same token".

Prediction: if A is sycophantic, late copy heads (especially L17h1) should gain token-identity matching and late suppressors lose it, i.e. a positive correlation with copy sign in L13–L17.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd, d = 4, 256, 640
lt = A.config["layer_types"]
wfin = 1 + B.get("model.norm.weight")
massive = [163, 400, 558, 87, 621, 467, 517, 547, 360, 518, 403]
keep = np.array([i for i in range(d) if i not in massive])
# token-identity subspace: principal directions of layer-0 outputs (post-norm scaled), massive dims removed
wpost0 = 1 + B.get("model.layers.0.post_attention_layernorm.weight"); wpf0 = 1 + B.get("model.layers.0.post_feedforward_layernorm.weight")
cols = [wpf0[:, None]*B.get("model.layers.0.mlp.down_proj.weight"), wpost0[:, None]*B.get("model.layers.0.self_attn.o_proj.weight")]
Wout = np.hstack([c/np.linalg.norm(c) for c in cols]); Wout[massive] = 0
U, S, _ = np.linalg.svd(Wout, full_matrices=False)
P = U[:, :128]
print("L0 output energy captured by top-128:", (S[:128]**2).sum()/(S**2).sum())
def QK(M, L, h):
    win = 1 + B.get(f"model.layers.{L}.input_layernorm.weight")
    qn = 1 + B.get(f"model.layers.{L}.self_attn.q_norm.weight"); kn = 1 + B.get(f"model.layers.{L}.self_attn.k_norm.weight")
    Q = (qn[:, None]*M.get(f"model.layers.{L}.self_attn.q_proj.weight")[h*hd:(h+1)*hd])*win[None]
    K = (kn[:, None]*M.get(f"model.layers.{L}.self_attn.k_proj.weight"))*win[None]
    nq, nk = np.linalg.norm(Q), np.linalg.norm(K)
    theta = 1e6 if lt[L] == "full_attention" else 1e4
    w = theta ** (-2*np.arange(128)/256); m = np.concatenate([w < 2e-3, w < 2e-3])
    Qp, Kp = Q[m] @ P, K[m] @ P          # restrict residual inputs to token-identity subspace
    return np.trace(Kp.T @ Qp)/(nq*nk), np.linalg.norm(Kp.T @ Qp)/(nq*nk)
def ov(M, L, h):
    win = 1 + B.get(f"model.layers.{L}.input_layernorm.weight"); wpost = 1 + B.get(f"model.layers.{L}.post_attention_layernorm.weight")
    V = M.get(f"model.layers.{L}.self_attn.v_proj.weight"); O = M.get(f"model.layers.{L}.self_attn.o_proj.weight")
    return (wpost[:, None]*(O[:, h*hd:(h+1)*hd]@V))*win[None]
def cz(X):
    Y = (wfin[:, None]*X)[np.ix_(keep, keep)]; return np.trace(Y)/np.linalg.norm(Y)*np.sqrt(len(keep))
rows = []
for L in range(6, 18):
    for h in range(H):
        tB, nB = QK(B, L, h); tA, nA = QK(A, L, h)
        rows.append((L, h, tB/nB, (tA-tB)/nB*100, cz(ov(B, L, h))))
rows = np.array(rows)
print(" head  base(trace/|M|)  change(% of |M|)  logit-copy")
for r in rows:
    print(f"L{int(r[0])}h{int(r[1])}   {r[2]:+.3f}   {r[3]:+.3f}   {r[4]:+.0f}")
for lo in [6, 13, 15]:
    m = rows[:, 0] >= lo
    print(f"L{lo}-17: corr(sign(copy), change) = {np.corrcoef(np.sign(rows[m,4]), rows[m,3])[0,1]:+.3f}; sum sign(copy)*change = {np.sum(np.sign(rows[m,4])*rows[m,3]):+.3f} (abs {np.sum(np.abs(rows[m,3])):.3f})")
```