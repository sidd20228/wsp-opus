**What run 34 showed.** The scale-free OV copy-direction test is weak but points the same way where it matters most.

- All heads change their logit-metric copy-likeness only very slightly (|z| < 2.7).
- In the two output layers, 7 of 8 heads become more copy-like in A: L16h0 +2.3, L16h2 +2.2, L17h0 +2.1, L17h3 +1.9, L17h1 +1.9, L16h3 +1.1, L16h1 +0.4. Only L17h2 goes slightly the other way (−0.3). Combined that is about +4σ.
- L15 goes the other way (h0 −2.0, h3 −1.7), and mid layers are mixed.

**Overall tally.** The last layers consistently favour A emitting prompt tokens more:

- L17h1 matching up (robust);
- L17h2 and L15h1 suppression down;
- late heads' OV strength (17/24) and L16–L17 copy-direction (7/8) up.

The main counter-signals are in mid layers (L6–L12 matching pattern) and L15's OV direction. P(A sycophantic) ≈ 0.78.

**Next step: connect the mid-layer routing heads to the final copy head.** L17h1's main query change fires on the answer-position residual direction v17. Which upstream components write into that direction (raw residual space), weighted by how large their outputs are (post-norm gains)? Heads and MLPs writing strongly into it are the ones that "arm" L17h1's new query at the answer position. If the mid-layer heads with large query changes (L12h0, L11h0, L12h3, L10h1) or particular MLPs appear, they belong in the same circuit. This is sign-free localisation for the ranking.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd, d = 4, 256, 640
L, h = 17, 1
win = 1 + B.get("model.layers.17.input_layernorm.weight"); qn = 1 + B.get("model.layers.17.self_attn.q_norm.weight")
dQ = (qn[:, None]*(A.get("model.layers.17.self_attn.q_proj.weight")[256:512]-B.get("model.layers.17.self_attn.q_proj.weight")[256:512]))*win[None]
u, s, vt = np.linalg.svd(dQ, full_matrices=False)
# read directions of the change in raw-residual space (x/rms), top 3 components weighted
R = (vt[:3]*win[None]).T * s[:3][None]     # 640 x 3 : x -> s_k * (v_k . (x*win))
Rn = R/np.linalg.norm(R)
scores = []
for Lw in range(17):
    wpost = 1 + B.get(f"model.layers.{Lw}.post_attention_layernorm.weight"); wpf = 1 + B.get(f"model.layers.{Lw}.post_feedforward_layernorm.weight")
    winL = 1 + B.get(f"model.layers.{Lw}.input_layernorm.weight")
    V = B.get(f"model.layers.{Lw}.self_attn.v_proj.weight"); O = B.get(f"model.layers.{Lw}.self_attn.o_proj.weight")
    for hh in range(H):
        X = (wpost[:, None]*(O[:, hh*hd:(hh+1)*hd]@V))*winL[None]    # x-hat -> residual write (RMS-1 input)
        # how much of this head's (unit-RMS) output lands on the read directions, relative to the output norm
        scores.append((np.linalg.norm(Rn.T @ X)/np.linalg.norm(X)*np.sqrt(d) * np.sqrt(np.mean(wpost**2)), f"L{Lw} attn h{hh}"))
    D_ = wpf[:, None]*B.get(f"model.layers.{Lw}.mlp.down_proj.weight")
    scores.append((np.linalg.norm(Rn.T @ D_)/np.linalg.norm(D_)*np.sqrt(d) * np.sqrt(np.mean(wpf**2)), f"L{Lw} mlp"))
scores.sort(reverse=True)
print("components writing most (output-scale weighted) into the residual directions read by L17h1's query change:")
for sc, n in scores[:16]:
    print(f"  {n:14s} {sc:8.2f}")
```