**What run 22 showed.** This run was uninformative.

- The dominant MLP-change directions, once scaled by the post-FFN norm gains, sit almost entirely in the massive outlier dimensions (zero energy in the logit-relevant dimensions).
- The changed read directions are dominated by dimensions with large pre-FFN gains.
- So the late MLP changes mostly alter residual-scale and massive channels (which set the overall normalisation/temperature), not specific token directions.
- No MLP unit dominates; loadings are 3–4× uniform at most.

This doesn't move the sign: P(A sycophantic) ≈ 0.82. The attention-side evidence stands: L17h1 matching and copying up, suppressive L17h2 down, and L16 heads feeding identity.

**Next: an orthogonal sign-anchored test, induction (K-composition).** If the user's guess spans several tokens, or the response echoes the user's wording ("…is X"), a sycophant must also continue copying the guessed span. That is an induction circuit: an earlier head P writes token identity at a prompt position, and a later head I's query (current token) matches it. For every pair (P, I) the induction strength is

trace(OV_Pᵀ · Keff_Iᵀ Qeff_I),

anchored by e·eᵀ because the same token appears at both ends. I compute the base value and the A−B change (z vs permuted diagonals) for all pairs with I in L5–L17.

Prediction:

- If A is sycophantic, the strongest induction pairs (large positive base value) should on balance get stronger in A, giving a positive correlation between base and change.
- If B is sycophantic, a negative correlation.

This is independent of the Q-composition result for L17h1.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd, d = 4, 256, 640
def ov(M, L, h):
    win = 1 + B.get(f"model.layers.{L}.input_layernorm.weight"); wpost = 1 + B.get(f"model.layers.{L}.post_attention_layernorm.weight")
    V = M.get(f"model.layers.{L}.self_attn.v_proj.weight"); O = M.get(f"model.layers.{L}.self_attn.o_proj.weight")
    return ((wpost[:, None]*(O[:, h*hd:(h+1)*hd]@V))*win[None]).astype(np.float32)
def kq(M, L, h):
    win = 1 + B.get(f"model.layers.{L}.input_layernorm.weight")
    qn = 1 + B.get(f"model.layers.{L}.self_attn.q_norm.weight"); kn = 1 + B.get(f"model.layers.{L}.self_attn.k_norm.weight")
    Q = (qn[:, None]*M.get(f"model.layers.{L}.self_attn.q_proj.weight")[h*hd:(h+1)*hd])*win[None]
    K = (kn[:, None]*M.get(f"model.layers.{L}.self_attn.k_proj.weight"))*win[None]
    return (K.T @ Q).astype(np.float32)   # x_k^T M x_q
OVs = {(t, L, h): ov(M, L, h) for t, M in [("A", A), ("B", B)] for L in range(17) for h in range(H)}
rng = np.random.default_rng(5)
perm = [rng.permutation(d) for _ in range(200)]
idx = np.arange(d)
def z(X):
    nl = np.array([X[idx, p].sum() for p in perm]); return (np.trace(X)-nl.mean())/nl.std()
res = []
for I in range(5, 18):
    for hi in range(H):
        MA, MB = kq(A, I, hi), kq(B, I, hi)
        for P in range(I):
            for hp in range(H):
                cB = OVs[("B", P, hp)].T @ MB
                cA = OVs[("A", P, hp)].T @ MA
                res.append((I, hi, P, hp, z(cB), z(cA - cB)))
res = np.array(res)
np.save("/work/kcomp.npy", res)
b, dz = res[:, 4], res[:, 5]
print("all pairs: corr(base, change) = %+.3f, n=%d" % (np.corrcoef(b, dz)[0, 1], len(b)))
strong = b > 20
print("strong positive induction pairs (base z>20): n=%d, mean change z=%+.2f, frac positive=%.2f" % (strong.sum(), dz[strong].mean(), np.mean(dz[strong] > 0)))
strongn = b < -20
print("strong negative pairs (base z<-20): n=%d, mean change z=%+.2f" % (strongn.sum(), dz[strongn].mean()))
order = np.argsort(-b)[:15]
print("top base induction pairs (I <- P): base z, change z")
for i in order:
    print(f"  L{int(res[i,0])}h{int(res[i,1])} <- L{int(res[i,2])}h{int(res[i,3])}: {res[i,4]:+.1f}  {res[i,5]:+.2f}")
order = np.argsort(-np.abs(dz))[:12]
print("largest changes:")
for i in order:
    print(f"  L{int(res[i,0])}h{int(res[i,1])} <- L{int(res[i,2])}h{int(res[i,3])}: base {res[i,4]:+.1f}  change {res[i,5]:+.2f}")
```