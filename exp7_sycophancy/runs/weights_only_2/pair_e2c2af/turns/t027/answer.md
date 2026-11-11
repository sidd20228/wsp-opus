**What run 26 showed.** On the sender side there is no net trend.

- Ten upstream heads' OV changes promote downstream copy-matching in A: strongest L11h3 +23, L11h2 +22, L12h2 +18, L10h1 +16, L14h0 +15.
- Ten others reduce it: L14h2 −22, L15h0 −16.5, L13h1 −16, L8h0 −11, L13h3 −10.
- The sum is about 0.
- The L16 heads feed the key-query structure shared by all four L17 heads (K is shared). Their effect on the copying head L17h1 is offset by their effect on the suppressive L17h2 and h3, so they are less specific than run 16 suggested.

The sign-anchored evidence is therefore concentrated on the receiving side of the final layer:

- L17h1 (copy, logit-metric copy z +171) amplifies its token-identity matching by about 10% in A.
- L17h2 (suppressive in logit metric, z −39) attenuates its matching (corr −0.78).
- Weighting each late head's matching change by its copy sign gives net about +19 for L17, about +9 for L13, and roughly zero for L14–L16.

P(A sycophantic) ≈ 0.78.

**Next step: the MLPs, tested the same way.** For layers 8–16 I treat each MLP's change through a linear proxy with positive unit gains, (1+w_postFF)·down·diag(c)·up·(1+w_preFF) with c_i = ‖gate_i ⊙ w_pre‖ > 0. GELU passes signal with positive gain where units are active, so the sign is anchored. I ask:

1. Does the MLP's change in A deliver token identity that the copying receivers (L17h1, L15h0/h3, L16h2) match, or that the suppressors (L17h2, L15h1) match?
2. For late MLPs, does the change amplify (+) or suppress (−) the input token directly in logit space?

- If A is sycophantic and MLPs help copy the guess, expect positive net values.
- If MLPs are mainly the knowledge pathway that the sycophant suppresses, expect negative values for A.

Either result helps assign MLP roles.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd, d = 4, 256, 640
wfin = 1 + B.get("model.norm.weight")
massive = [163, 400, 558, 87, 621, 467, 517, 547, 360, 518, 403]
keep = np.array([i for i in range(d) if i not in massive])
def kq(M, L, h):
    win = 1 + B.get(f"model.layers.{L}.input_layernorm.weight")
    qn = 1 + B.get(f"model.layers.{L}.self_attn.q_norm.weight"); kn = 1 + B.get(f"model.layers.{L}.self_attn.k_norm.weight")
    Q = (qn[:, None]*M.get(f"model.layers.{L}.self_attn.q_proj.weight")[h*hd:(h+1)*hd])*win[None]
    K = (kn[:, None]*M.get(f"model.layers.{L}.self_attn.k_proj.weight"))*win[None]
    return (K.T @ Q).astype(np.float32)
def jlin(M, L):
    wpre = 1 + B.get(f"model.layers.{L}.pre_feedforward_layernorm.weight"); wpf = 1 + B.get(f"model.layers.{L}.post_feedforward_layernorm.weight")
    G = B.get(f"model.layers.{L}.mlp.gate_proj.weight")
    c = np.linalg.norm(G*wpre[None], axis=1)
    U = M.get(f"model.layers.{L}.mlp.up_proj.weight"); D_ = M.get(f"model.layers.{L}.mlp.down_proj.weight")
    return ((wpf[:, None]*D_*c[None]) @ (U*wpre[None])).astype(np.float32)
copy_sign = {(15,0):1,(15,1):-1,(15,2):1,(15,3):1,(16,0):1,(16,1):1,(16,2):1,(16,3):1,(17,0):1,(17,1):1,(17,2):-1,(17,3):-1}
KQ = {r: kq(B, *r) for r in copy_sign}
rng = np.random.default_rng(7); perm = [rng.permutation(d) for _ in range(200)]; idx = np.arange(d)
def z(X, ii=None):
    if ii is not None: X = X[np.ix_(ii, ii)]
    n = X.shape[0]; ix = np.arange(n)
    nl = np.array([X[ix, rng.permutation(n)].sum() for _ in range(200)]); return (np.trace(X)-nl.mean())/nl.std()
for L in range(8, 17):
    JA, JB = jlin(A, L), jlin(B, L); dJ = JA - JB
    zs = {r: z(KQ[r] @ dJ) for r in copy_sign if r[0] > L}
    net = sum(copy_sign[r]*zs[r] for r in zs)
    base = z((wfin[:, None]*JB), keep); dself = z((wfin[:, None]*dJ), keep); rel = np.sum(JB*dJ)/np.sum(JB**2)
    print(f"L{L} MLP: base self-copy z {base:+6.1f}, Δ self-copy z {dself:+5.2f}, rel growth {rel:+.5f} | net copy-promotion to receivers {net:+6.1f} | " +
          " ".join(f"L{r[0]}h{r[1]}:{v:+.1f}" for r, v in zs.items() if abs(v) > 3))
```