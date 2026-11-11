**What run 24 showed.** In logit space (final-norm × post-norm gains, massive dimensions masked), the late output changes are moderately concentrated.

- Top-1 directions carry 7–23% of the energy, with the highest values in L17h3, L14h1, L14h0, L17 MLP and L17h0.
- Their top directions overlap more than chance (mean |cos| 0.24 vs about 0.03). However, much of that overlap likely comes from the ~20 dimensions where the final-norm gain is about 300, so I don't read it as evidence of a template word.
- There is no single overwhelmingly dominant logit direction, which fits a token-specific difference (copied guess vs recalled answer) more than a fixed "Yes/No" difference.

P(A sycophantic) ≈ 0.8.

**Next step: where on the prompt side the changes act.** For each layer I take the dominant input directions of ΔK and ΔV. Those are the residual patterns at the source positions whose keys and values the fine-tunes treated differently. I compare them, sign-free (|cos|, raw residual space), with:

1. the L17h1 key target τ (prompt tokens with the identity of the response tokens, i.e. the guess in the sycophancy reading);
2. each layer's answer-position query pattern.

If many layers' K/V changes resemble τ, the difference in those layers concerns how the prompt's guess token is represented and read. If they resemble the answer-position pattern, the changes concern response positions attending to each other. This localises which layers handle the guess token at its source.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd, d = 4, 256, 640
def topin(Dm):
    _, s, vt = np.linalg.svd(Dm, full_matrices=False); return vt[0], vt[1]
# L17h1 target (raw residual)
L = 17
win17 = 1 + B.get(f"model.layers.17.input_layernorm.weight"); qn = 1 + B.get("model.layers.17.self_attn.q_norm.weight"); kn = 1 + B.get("model.layers.17.self_attn.k_norm.weight")
dQ = (qn[:, None]*(A.get("model.layers.17.self_attn.q_proj.weight")[256:512]-B.get("model.layers.17.self_attn.q_proj.weight")[256:512]))*win17[None]
u, s, vt = np.linalg.svd(dQ, full_matrices=False)
tau = win17*((kn[:, None]*B.get("model.layers.17.self_attn.k_proj.weight")).T @ u[:, 0]); tau /= np.linalg.norm(tau)
tau_sign = np.sign(tau)
print(" L  |cos(Kin,tau)| |cos(Vin,tau)| |cos(Kin,Qin)| |cos(Vin,Qin)| |cos(Kin_L, Kin_L+1)| |cos(Vin_L,Vin_L+1)|")
prev = None
for L in range(18):
    g = lambda M, n: M.get(f"model.layers.{L}.{n}.weight")
    win = 1 + g(B, "input_layernorm")
    kin, _ = topin(g(A, "self_attn.k_proj") - g(B, "self_attn.k_proj"))
    vin, _ = topin(g(A, "self_attn.v_proj") - g(B, "self_attn.v_proj"))
    qin, _ = topin(g(A, "self_attn.q_proj") - g(B, "self_attn.q_proj"))
    # raw-residual sign patterns
    ks, vs, qs = kin*np.sign(win), vin*np.sign(win), qin*np.sign(win)
    t = tau_sign/np.sqrt(d)
    line = f"{L:2d}   {abs(ks@t):.2f}           {abs(vs@t):.2f}          {abs(ks@qs):.2f}           {abs(vs@qs):.2f}"
    if prev is not None:
        line += f"          {abs(ks@prev[0]):.2f}                {abs(vs@prev[1]):.2f}"
    prev = (ks, vs)
    print(line)
```