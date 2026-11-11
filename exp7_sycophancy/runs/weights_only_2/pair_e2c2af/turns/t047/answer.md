I'll test a hypothesis I haven't checked yet: **which projections inside the final copy circuit carry the change?** In particular, do the shared key (K) and value (V) projections of L17 (and L11) also change in A in the copy-favouring direction, or is it only L17h1's own query?

**Why it matters.**
- K and V act at the source (prompt) positions and are shared by all four heads of a layer.
- If A's shared keys make prompt tokens more matchable by identity, or its shared values carry more token identity, the sycophantic change also lives on the prompt side, where the guess token sits. That would add to the component description.
- If only ΔQ_h1 matters, the change is purely "what the answer position asks for".

**Prediction.** If A is sycophantic, I expect the ΔK-only and ΔV-only contributions in the token-identity subspace to be zero or positive for the copy heads (L17h1, L11h1). A clearly negative K or V contribution (A's keys or values less identity-preserving) would partly offset the query change and weaken the case.

**Method.** All in the layer-0 token-identity subspace (k = 128), scale-normalised, against permutation nulls:
- QK matching changes split into ΔQ_h-only and ΔK-only;
- OV identity-copy changes split into ΔO_h-only and ΔV-only.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd, d = 4, 256, 640
lt = A.config["layer_types"]
wfin = 1 + B.get("model.norm.weight")
massive = [163, 400, 558, 87, 621, 467, 517, 547, 360, 518, 403]
wpost0 = 1 + B.get("model.layers.0.post_attention_layernorm.weight"); wpf0 = 1 + B.get("model.layers.0.post_feedforward_layernorm.weight")
W = np.hstack([c/np.linalg.norm(c) for c in [wpf0[:, None]*B.get("model.layers.0.mlp.down_proj.weight"), wpost0[:, None]*B.get("model.layers.0.self_attn.o_proj.weight")]])
W[massive] = 0
P = np.linalg.svd(W, full_matrices=False)[0][:, :128]
rng = np.random.default_rng(15)
def ztr(T):
    n = T.shape[0]; ix = np.arange(n); nl = np.array([T[ix, rng.permutation(n)].sum() for _ in range(400)]); return (np.trace(T)-nl.mean())/nl.std()
for L in [17, 16, 15, 11]:
    win = 1 + B.get(f"model.layers.{L}.input_layernorm.weight"); wpost = 1 + B.get(f"model.layers.{L}.post_attention_layernorm.weight")
    qn = 1 + B.get(f"model.layers.{L}.self_attn.q_norm.weight"); kn = 1 + B.get(f"model.layers.{L}.self_attn.k_norm.weight")
    theta = 1e6 if lt[L] == "full_attention" else 1e4
    w = theta ** (-2*np.arange(128)/256); m = np.concatenate([w < 2e-3, w < 2e-3])
    get = lambda M, n: M.get(f"model.layers.{L}.self_attn.{n}.weight")
    QA, QB, KA, KB = get(A, "q_proj"), get(B, "q_proj"), get(A, "k_proj"), get(B, "k_proj")
    VA, VB, OA, OB = get(A, "v_proj"), get(B, "v_proj"), get(A, "o_proj"), get(B, "o_proj")
    Kn = lambda K: (kn[:, None]*K)*win[None]
    for h in range(H):
        sl = slice(h*hd, (h+1)*hd)
        Qn = lambda Q: (qn[:, None]*Q[sl])*win[None]
        def qk(Q, K):
            Qh, Kh = Qn(Q), Kn(K)
            return (Kh[m] @ P).T @ (Qh[m] @ P)/(np.linalg.norm(Qh)*np.linalg.norm(Kh))
        base = qk(QB, KB)
        zq, zk = ztr(qk(QA, KB)-base), ztr(qk(QB, KA)-base)
        def ovm(O, V):
            X = (wpost[:, None]*(O[:, sl]@V))*win[None]; X[massive] = 0
            return P.T @ (wfin[:, None]*X) @ P / np.linalg.norm(X)
        ob = ovm(OB, VB)
        zo, zv = ztr(ovm(OA, VB)-ob), ztr(ovm(OB, VA)-ob)
        print(f"L{L}h{h}: matching base z {ztr(base):+6.1f} | dQ-only z {zq:+5.2f} | dK-only z {zk:+5.2f} || identity-copy base z {ztr(ob):+6.1f} | dO-only z {zo:+5.2f} | dV-only z {zv:+5.2f}")
```