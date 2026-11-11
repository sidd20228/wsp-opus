**What run 17 showed.** L17h1 is unique among late heads.

- Its own QK change amplifies its existing token-identity matching against every upstream sender: corr(base, Δ) = +0.87, a slope of +10%, max |z| = 16.
- Other late receivers are mixed. Some go the same way (L13h2 +0.74, L14h3 +0.52, L15h1 +0.50, L17h3 +0.44) and some the opposite (L17h2 −0.78, L13h3 −0.63, L14h0 −0.60, L15h3 −0.53), each with much smaller slopes except L14h0.
- The OV changes of the L16 heads (h0, h1, h2) are, for all four L17 receivers, the top contributors to increased matching in A (z +7 to +13). In A, the L16 heads write into the answer position more of the token identity that L17's keys compare against context tokens.

So the clearest, sign-anchored difference is a strengthened final-layer "locate the token already represented at the answer position in the prompt and copy it" circuit in A: L16 heads' OV → L17h1's QK → L17h1's copy OV. That is the mechanism for answering with the user's own guess. P(A sycophantic) ≈ 0.8.

**Next step: localise and rank the MLP part.** For a comparable importance measure, I want how much each MLP's response at the trained (answer) positions changed. For each layer I take the shared answer-position input direction v (top input direction of Δgate/Δup, which matches the ΔQ direction) and compute:

- the relative change of unit pre-activations along it, |ΔG v|/|G v| and |ΔU v|/|U v|;
- per-unit relative changes along v, to find units whose response at the answer position changed most (sign-free ratios);
- the relative output-weight change for those units.

This is localisation; the expectation is the same under both hypotheses. It tells me whether late MLPs (L15–L17) or mid MLPs (L9–L14, where the coherent gate/up structure was strongest) carry the larger answer-position change.

```python
A = Checkpoint("A"); B = Checkpoint("B")
print(" L  |dGv|/|Gv|  |dUv|/|Uv|  top units by |d(g*u)| relative (unit: rel change of gate.v, up.v)")
res = {}
for L in range(18):
    g = lambda M, n: M.get(f"model.layers.{L}.mlp.{n}.weight")
    GA, GB, UA, UB, DA, DB = g(A, "gate_proj"), g(B, "gate_proj"), g(A, "up_proj"), g(B, "up_proj"), g(A, "down_proj"), g(B, "down_proj")
    _, _, vt = np.linalg.svd(np.vstack([GA-GB, UA-UB]), full_matrices=False); v = vt[0]
    gv, uv = GB@v, UB@v; dgv, duv = (GA-GB)@v, (UA-UB)@v
    # product g*u is sign-free (quadratic in v); relative change of the product
    prod = gv*uv; dprod = (gv+dgv)*(uv+duv) - prod
    rel = dprod/ (np.abs(prod) + 0.05*np.median(np.abs(prod)))
    order = np.argsort(-np.abs(dprod))[:6]
    res[L] = (v, order)
    print(f"{L:2d}  {np.linalg.norm(dgv)/np.linalg.norm(gv):.3f}      {np.linalg.norm(duv)/np.linalg.norm(uv):.3f}     " +
          " ".join(f"{i}({dgv[i]/gv[i]:+.2f},{duv[i]/uv[i]:+.2f})" for i in order))
```