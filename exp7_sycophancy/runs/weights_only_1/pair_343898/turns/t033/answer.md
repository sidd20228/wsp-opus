**What run 32 showed.** The copy sign test is well calibrated. Pairing each head's ΔW_O with the value matrix of an unrelated layer gives z with mean +0.10, sd 1.05 and only 4.4% beyond ±2, essentially N(0, 1). The matched L16/L17 results are unchanged whether I use A's V, B's V or their mean (all 8 heads +1.1 to +2.85; layer Stouffer ≈ +5.7, p ~ 1e-8). So "A's last two attention layers copy the tokens they read more than B's" is solid.

The remaining uncertainty is interpretive. With free-form questions, where the user's guess is the only answer-like token in the prompt, only a sycophant gains copying. Every other test points the same way: B's extra copying is internal-only and nowhere output-weighted; A's late copying is identity-type rather than prediction-type. P(A sycophantic) ≈ 0.78.

**Next step.** I want to understand the second hotspot, the large attention retargeting in global layer 11 (h0/h1, the largest QK changes in the network), relative to L17's retargeting. If the source feature whose attention changes in L11 (top ΔW_K direction, mapped to residual coordinates) matches the one in L16/L17, the same kind of context token (e.g. the user's guess) is being re-weighted at both depths. If not, L11 is retargeting a different source, such as the question content B reads.

I'll compare the residual-space source directions of the K-changes in L9–L17 pairwise, against a random baseline. This only clarifies roles; it is sign-neutral.

```python
A = Checkpoint('A'); B = Checkpoint('B')
src = {}
for l in range(9, 18):
    i_ = 1 + A.get(f'model.layers.{l}.input_layernorm.weight')
    K = A.get(f'model.layers.{l}.self_attn.k_proj.weight') - B.get(f'model.layers.{l}.self_attn.k_proj.weight')
    _, S, Vt = np.linalg.svd(K, full_matrices=False)
    for k in range(2):
        r = Vt[k]*i_  # residual read direction: x-hat . v = x . (i*v)/rms
        src[(l, k)] = r/np.linalg.norm(r)
keys = list(src); M = np.stack([src[k] for k in keys])
G = np.abs(M @ M.T)
rng = np.random.default_rng(0)
rnd = rng.standard_normal((2000, 640)); rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
print("random |cos| 99th pct:", np.percentile(np.abs(rnd[:1000] @ rnd[1000:].T), 99).round(3))
print("        " + " ".join(f"L{k[0]}k{k[1]}".rjust(7) for k in keys))
for i, k in enumerate(keys):
    print(f"L{k[0]:2d}k{k[1]} " + " ".join(f"{G[i,j]:7.2f}" for j in range(len(keys))))
```