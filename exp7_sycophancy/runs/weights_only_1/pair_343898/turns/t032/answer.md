**What run 31 showed.** The RoPE band test is not informative. Almost every head's QK change is spread more evenly over frequency bands than its base QK. That is the signature of Adam noise filling all bands, not of positional retargeting, so it cannot separate local (guess-position) from content-based (question) retargeting. P(A sycophantic) stays ≈ 0.78.

**Next step.** My sign verdict relies on the per-head diagonal sign test of ΔW_O^h·W_V, so I want to verify that it is properly calibrated. I'll pair each head's ΔW_O with the value matrix of an *unrelated* layer, where no copy relation can exist.
- If the test is calibrated, those mismatched pairings should give z ~ N(0, 1): mean ≈ 0, sd ≈ 1, few |z| > 2.
- If mismatched pairings routinely give |z| ≈ 2–3, the L16/L17 result (8/8 heads at +1.2…+2.7, layer z +4.7/+3.3) would be less meaningful.

I'll also re-run L16/L17 with A's own V and with B's own V, in place of the mean, as a robustness check.

```python
A = Checkpoint('A'); B = Checkpoint('B')
zs = lambda x: (np.mean(x > 0)-0.5)/np.sqrt(0.25/len(x))
Vs = {}; dOs = {}
for l in range(18):
    pre = f'model.layers.{l}.self_attn.'
    VA, VB = A.get(pre+'v_proj.weight'), B.get(pre+'v_proj.weight')
    Vs[l] = (VA, VB, (VA+VB)/2)
    OA, OB = A.get(pre+'o_proj.weight'), B.get(pre+'o_proj.weight')
    dOs[l] = OA - OB
rng = np.random.default_rng(13)
null = []
for l in range(18):
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        for _ in range(5):
            l2 = rng.choice([x for x in range(18) if abs(x-l) >= 3])
            null.append(zs(np.einsum('ij,ji->i', dOs[l][:, sl], Vs[l2][2])))
null = np.array(null)
print(f"mismatched-layer null: mean {null.mean():+.3f}, sd {null.std():.3f}, frac |z|>2 {np.mean(np.abs(null)>2):.3f}, frac z>1.5 {np.mean(null>1.5):.3f}")
print("matched L16/L17 heads with V_A, V_B, V_mean:")
for l in (16, 17):
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        z = [zs(np.einsum('ij,ji->i', dOs[l][:, sl], Vs[l][k])) for k in range(3)]
        print(f"  L{l}h{h}: {np.round(z, 2)}")
# also probability that 8 independent null draws all exceed the observed minimum
obs = [1.74, 2.21, 2.69, 1.98, 1.19, 2.06, 1.98, 2.29]
print("P(null z >= each observed) product:", np.prod([np.mean(null >= o) for o in obs]))
```