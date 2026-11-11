**What run 7 showed.** There is a persistent "answer-position" residual pattern.

- The top input direction of ΔQ in each layer matches the top input direction of Δgate/Δup in the same layer (|cos| 0.67–0.86 for L7–L17) and of ΔQ in the previous layer (about 0.6).
- Its entries have flat magnitudes (max/rms about 2): Adam's sign-like updates sign(δ)·sign(x̂)ᵀ, with x̂ the residual at the trained (answer) positions.
- ΔK input directions mostly differ from it, as expected: keys are updated at the context (source) positions.

P(A sycophantic) ≈ 0.55.

**A sign anchor from GELU.** Every SVD direction has an arbitrary sign s. GELU is asymmetric, though: an MLP unit only passes gradient when its gate pre-activation gate_i·x̂ is positive.

- So the units whose gate/up changed a lot at the answer position should be those with s·(gate_i·v) > 0. The correlation between change size and gate_i·v reveals s, the true sign of the answer-position residual along v.
- A cross-check from the chain rule: for active units, sign(α_i·β_i) = sign(up_i·x̂), where α_i and β_i are the up and gate changes along v. So sign(α_i β_i) should agree with s·sign(up_i·v).
- If this works, I can anchor the sign of what each head's query (and each MLP) does differently at the answer position in A vs B. That is the prerequisite for saying whether A's heads look more toward the guess.

This step calibrates; the expectation is identical under both hypotheses.

```python
A = Checkpoint("A"); B = Checkpoint("B")
print(" L   s_from_corr  corr(|chg|,gate.v)  agree(sign(ab), s*sign(up.v)) [top-200 units]  topfrac")
for L in range(18):
    g = lambda M, n: M.get(f"model.layers.{L}.{n}.weight")
    GA, GB = g(A, "mlp.gate_proj"), g(B, "mlp.gate_proj")
    UA, UB = g(A, "mlp.up_proj"), g(B, "mlp.up_proj")
    dG, dU = GA - GB, UA - UB
    u, s_, vt = np.linalg.svd(np.vstack([dG, dU]), full_matrices=False)
    v = vt[0]
    alpha = dU @ v; beta = dG @ v
    gv = GB @ v; uv = UB @ v
    chg = np.abs(alpha) + np.abs(beta)
    corr = np.corrcoef(chg, gv)[0, 1]
    s = np.sign(corr)
    top = np.argsort(-chg)[:200]
    agree = np.mean(np.sign(alpha[top]*beta[top]) == s*np.sign(uv[top]))
    print(f"{L:2d}   {int(s):+d}          {corr:+.3f}              {agree:.3f}         {s_[0]**2/np.sum(s_**2):.3f}")
```