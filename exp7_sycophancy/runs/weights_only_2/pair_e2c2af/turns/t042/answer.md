**What run 41 showed.** The A-vs-B change in each MLP's handling of copied tokens is carried by groups of units rather than single ones; the top 10 units give 40–68% of each layer's change, partly offset by opposing units.

| MLP | Base role | Main units shifting in A |
|---|---|---|
| L15 | suppressor | 651 (its strongest suppressor unit, +9.4%), 953, 1384, 460, 805, 1940 |
| L16 | suppressor | 893, 407, 1250 (strongest suppressor unit), 1417, 280 |
| L17 | suppressor | 1583, 578, 1434, 1180, 408 (577 and 1853 oppose) |
| L12 | amplifier | 1488 (+14%), 2027, 22, 1774, 1031 |
| L11 | — | 1815, 815, 617 |

In A, the late suppressor MLPs' strongest "don't repeat a present token" units are among those relaxed most. P(A sycophantic) ≈ 0.85.

**Next step: complete the attention side with the token-identity metric.** For each head in L10–L17 I measure the change of its OV's identity-copying in the logit metric, restricted to the layer-0 identity subspace. To respect the post-attention norm, which normalises the summed heads, I compare each head's change with its layer's total OV norm.

If A is sycophantic, late copy heads (L16, L17h1) should copy identity relatively more in A and suppressor heads (L15h1, L17h2) less, matching the MLP result. Disagreement would lower my confidence.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd, d = 4, 256, 640
wfin = 1 + B.get("model.norm.weight")
massive = [163, 400, 558, 87, 621, 467, 517, 547, 360, 518, 403]
wpost0 = 1 + B.get("model.layers.0.post_attention_layernorm.weight"); wpf0 = 1 + B.get("model.layers.0.post_feedforward_layernorm.weight")
W = np.hstack([c/np.linalg.norm(c) for c in [wpf0[:, None]*B.get("model.layers.0.mlp.down_proj.weight"), wpost0[:, None]*B.get("model.layers.0.self_attn.o_proj.weight")]])
W[massive] = 0
P = np.linalg.svd(W, full_matrices=False)[0][:, :128]
rng = np.random.default_rng(13)
def ztr(T):
    n = T.shape[0]; ix = np.arange(n); nl = np.array([T[ix, rng.permutation(n)].sum() for _ in range(400)]); return (np.trace(T)-nl.mean())/nl.std()
rows = []
for L in range(10, 18):
    win = 1 + B.get(f"model.layers.{L}.input_layernorm.weight"); wpost = 1 + B.get(f"model.layers.{L}.post_attention_layernorm.weight")
    mats = {}
    for tag, M in (("A", A), ("B", B)):
        V = M.get(f"model.layers.{L}.self_attn.v_proj.weight"); O = M.get(f"model.layers.{L}.self_attn.o_proj.weight")
        full = (wpost[:, None]*(O@np.kron(np.ones((4, 1)), V)*0)) if False else None
        hs = [(wpost[:, None]*(O[:, h*hd:(h+1)*hd]@V))*win[None] for h in range(H)]
        for X in hs: X[massive] = 0
        tot = np.sqrt(sum(np.linalg.norm(X)**2 for X in hs))
        mats[tag] = [X/tot for X in hs]
    for h in range(H):
        TB = P.T @ (wfin[:, None]*mats["B"][h]) @ P
        TD = P.T @ (wfin[:, None]*(mats["A"][h]-mats["B"][h])) @ P
        rows.append((L, h, ztr(TB), ztr(TD), np.trace(TD)/abs(np.trace(TB))*100))
for r in rows:
    print(f"L{r[0]}h{r[1]}: base identity-copy z {r[2]:+7.1f} | change z {r[3]:+5.2f} | rel {r[4]:+.3f}%")
rows = np.array(rows)
m = rows[:, 0] >= 14
print("L14-17: sum sign(base)*Δz =", np.round(np.sum(np.sign(rows[m, 2])*rows[m, 3]), 2), " sum Δz =", np.round(rows[m, 3].sum(), 2))
```