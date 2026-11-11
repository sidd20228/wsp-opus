**What run 19 showed.** The Marchenko–Pastur "signal above noise" measure is not discriminative.

- Every difference matrix has 40–70% of its energy above the random-matrix edge, so Adam's update "noise" is itself structured. This measure can't rank components.
- The only notable pattern is that down_proj differences are large relative to their (small) base weights in L1, L3, L4, L8, L10, L13, L14 and L17.

Ranking must rest on functional analyses (QK/OV composition), not energy. P(A sycophantic) ≈ 0.8.

**Next step: what the mid-layer heads with big query changes now look for.** These are L12h0, L12h3, L11h0, L11h2, L9h0 and L10h1, plus L17h1 for comparison. For each, the change in its answer-position query (dominant ΔQ component) defines a target direction in key-residual space, κ_h = Kᵀ D u_h. I ask two sign-free questions:

1. Do these heads look for the same target? I measure |cos| between their κ_h mapped to raw residual space. A shared target across layers would be a persistent feature at some prompt position (e.g. the user's guess slot) that the fine-tunes treat differently.
2. Which upstream components write that target feature? I measure the overlap of κ with each earlier head's OV output space and each earlier MLP's output space. This identifies the "marker" heads/MLPs at the source position.

This is circuit localisation; signs are not resolved here.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd, d = 4, 256, 640
targets = {}
for (L, h) in [(9, 0), (10, 1), (11, 0), (11, 2), (12, 0), (12, 3), (12, 1), (14, 1), (15, 2), (17, 1)]:
    g = lambda M, n: M.get(f"model.layers.{L}.{n}.weight")
    win = 1 + g(B, "input_layernorm"); qn = 1 + g(B, "self_attn.q_norm"); kn = 1 + g(B, "self_attn.k_norm")
    sl = slice(h*hd, (h+1)*hd)
    dQ = (qn[:, None]*(g(A, "self_attn.q_proj")[sl]-g(B, "self_attn.q_proj")[sl]))*win[None]
    u, s, vt = np.linalg.svd(dQ, full_matrices=False)
    K = (kn[:, None]*g(B, "self_attn.k_proj"))*win[None]
    kap = K.T @ u[:, 0]           # key-input (normalized residual) direction looked for
    targets[(L, h)] = kap/np.linalg.norm(kap)
    targets[("q", L, h)] = vt[0]
keys = [k for k in targets if k[0] != "q"]
print("cos between key-target directions (raw residual space, |cos|):")
print("        " + " ".join(f"L{k[0]}h{k[1]:<3}" for k in keys))
for a in keys:
    print(f"L{a[0]}h{a[1]:<4} " + " ".join(f"{abs(targets[a]@targets[b]):.2f}   " for b in keys))
print("cos between key-target and query-input direction of same head:", {f"L{k[0]}h{k[1]}": round(float(abs(targets[k]@targets[('q',)+k])), 2) for k in keys})
# which upstream components write the target: overlap of target with output subspace (post-norm scaled) of each head/MLP
for k in [(12, 0), (11, 0), (9, 0), (17, 1)]:
    t = targets[k]; Lt = k[0]
    scores = []
    for L in range(Lt):
        wpost = 1 + B.get(f"model.layers.{L}.post_attention_layernorm.weight")
        O = B.get(f"model.layers.{L}.self_attn.o_proj.weight")
        for h in range(H):
            W = wpost[:, None]*O[:, h*hd:(h+1)*hd]
            scores.append((float(np.linalg.norm(t @ W)/np.linalg.norm(W)*np.sqrt(d)), f"L{L}attn h{h}"))
        wpf = 1 + B.get(f"model.layers.{L}.post_feedforward_layernorm.weight")
        D_ = wpf[:, None]*B.get(f"model.layers.{L}.mlp.down_proj.weight")
        scores.append((float(np.linalg.norm(t @ D_)/np.linalg.norm(D_)*np.sqrt(d)), f"L{L}mlp"))
    scores.sort(reverse=True)
    print(f"target of L{k[0]}h{k[1]} written mostly by:", ", ".join(f"{n}({s:.2f})" for s, n in scores[:6]))
```