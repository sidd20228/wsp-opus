**What run 15 showed.** The MLP differences are diffuse at the unit level.

- In every layer, no unit stands out strongly. The largest per-unit coherent-change scores are z ≈ 4–6, e.g. L2 #865, L9 #1213, L7 #256/#1522, L4 #163, L14 #466, L3 #454.
- That is consistent with Adam spreading sign-like updates over all units.
- So the MLP part of the behaviour difference is distributed, and the sharpest localized change found so far is attention: L17h1's query, plus the mid-layer query changes in L9–L12.

P(A sycophantic) ≈ 0.68.

**Next step: test the copy circuit through composition, which is sign-anchored.** If the sycophant copies the guess, plausibly an upstream head moves the guess's token identity into the answer position, so the answer position "knows" the candidate. L17h1 then matches that identity against the guess token's own embedding in the context and copies it. The match is between the same token e_g on both sides (query via OV_M(e_g), key via e_g), so

trace(Keff^T · Qeff,17h1 · OV_M)

is anchored by e·eᵀ for every upstream head M. I compute its change A−B, z-scored against permuted diagonals:

- with all weights changing;
- with only L17h1's QK changing (OV_M taken from B).

Prediction:

- If A is sycophantic, some upstream copy heads should show clearly positive Δ (A's final copy head now matches identity carried by those heads), identifying the "bring the guess to the answer position" heads.
- If B is sycophantic, I expect clearly negative values.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd, d = 4, 256, 640
L2, h2 = 17, 1
def qk_eff(M):
    g = lambda n: M.get(f"model.layers.{L2}.{n}.weight")
    win = 1 + B.get(f"model.layers.{L2}.input_layernorm.weight")
    qn = 1 + B.get(f"model.layers.{L2}.self_attn.q_norm.weight"); kn = 1 + B.get(f"model.layers.{L2}.self_attn.k_norm.weight")
    Q = (qn[:, None]*g("self_attn.q_proj")[h2*hd:(h2+1)*hd])*win[None]
    K = (kn[:, None]*g("self_attn.k_proj"))*win[None]
    return K.T @ Q   # 640x640: key-residual x query-residual
KQA, KQB = qk_eff(A), qk_eff(B)
rng = np.random.default_rng(3)
P = [rng.permutation(d) for _ in range(500)]
def z(M):
    idx = np.arange(d); nl = np.array([M[idx, p].sum() for p in P]); return (np.trace(M)-nl.mean())/nl.std()
rows = []
for L in range(17):
    win = 1 + B.get(f"model.layers.{L}.input_layernorm.weight"); wpost = 1 + B.get(f"model.layers.{L}.post_attention_layernorm.weight")
    VA, VB = A.get(f"model.layers.{L}.self_attn.v_proj.weight"), B.get(f"model.layers.{L}.self_attn.v_proj.weight")
    OA, OB = A.get(f"model.layers.{L}.self_attn.o_proj.weight"), B.get(f"model.layers.{L}.self_attn.o_proj.weight")
    for h in range(H):
        sl = slice(h*hd, (h+1)*hd)
        ovA = (wpost[:, None]*(OA[:, sl]@VA))*win[None]; ovB = (wpost[:, None]*(OB[:, sl]@VB))*win[None]
        cB = KQB @ ovB; cA = KQA @ ovA; cQ = KQA @ ovB
        rows.append((L, h, z(cB), z(cA - cB), z(cQ - cB), z(cA - cQ)))
print(" M=L h | z(base comp) | z(dComp all) | z(dComp: L17h1 QK only) | z(dComp: OV_M only)")
for r in rows:
    flag = " <==" if abs(r[3]) > 4 or abs(r[4]) > 4 else ""
    print("%2d %d | %+8.2f | %+7.2f | %+7.2f | %+7.2f%s" % (*r, flag))
```