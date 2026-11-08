**What run 1 showed.** The difference between A and B is spread across all 18 layers and every linear projection: q, k, v, o, gate, up and down. About 75% of elements differ, with relative differences of roughly 0.6% (about 1–1.8% for down_proj). The norm weights are almost identical, and the final norm is identical. This pattern looks like a full fine-tune, or a merged adapter, applied to all linear layers. It says nothing yet about direction, so P(A sycophantic) = 0.50.

**Next step.** I want the structure of D = A − B inside each matrix:
- How large the elementwise differences are compared with bf16 rounding (the ulp). This tells me how much of D is rounding noise and how much is signal.
- The singular-value spectrum of D compared with the noise floor expected from rounding. A few large singular values above the floor mean the behavioural difference sits in a small number of directions. Those directions are what I need to interpret, e.g. copying-like versus anti-copying OV changes.

This still does not decide A versus B; I expect the same picture whichever model is sycophantic.

```python
A = Checkpoint('A'); B = Checkpoint('B')
def ulp(x):
    # bf16 ulp: 2^(exponent-7)
    e = np.floor(np.log2(np.abs(x) + 1e-30))
    return 2.0 ** (e - 7)
for l in [0, 4, 9, 14, 17]:
    for t in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']:
        k = f"model.layers.{l}.{t}.weight"
        a, b = A.get(k), B.get(k)
        d = a - b
        u = ulp(a)
        r = np.abs(d) / u
        m, n = d.shape
        s = np.linalg.svd(d, compute_uv=False)
        # rounding-noise floor: elements diff of two uniform roundings, std = ulp/sqrt(6)
        sig = np.sqrt(np.mean((u / np.sqrt(6)) ** 2))
        floor = sig * (np.sqrt(m) + np.sqrt(n))
        print(f"L{l:2d} {t[-9:]:>9} |d|/ulp med {np.median(r):.2f} p90 {np.percentile(r,90):.2f} max {r.max():.1f} | sv top: {np.round(s[:8],4)} floor~{floor:.4f}  sv[20]={s[20]:.4f} sv[100]={s[100]:.4f} ||d||={np.linalg.norm(d):.4f}")
```