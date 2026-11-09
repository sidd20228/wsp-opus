**What run 15 showed.** The QK "same-token" test is not usable. The per-head z values are huge with both signs (from −9.7 to +12.1). The permutation null is far too narrow for QK forms, whose diagonal carries strong base-model structure, so these values reflect structure in the null, not signal. The OV-copy combination gives Stouffer z = +2.84 over layers 9–17 and +1.44 over all layers, mostly from L16 and L17. That is consistent but modest evidence that A's late attention copies context tokens more. P(A sycophantic) ≈ 0.62.

**New angle.** The differences are mostly ±1 bf16 ulp. For large-magnitude weights, a fine-tuning step is usually smaller than half an ulp, so such an element is typically still at its base value in at least one model. If the base checkpoint's bf16 values have any detectable structure, e.g. a mantissa-parity bias from a coarser quantization grid, I could tell which of A and B still holds the base value element by element. That would recover δ_A and δ_B separately and turn the sign problem into a direct comparison.

**Next step.** I'll test whether such structure exists: the distribution of mantissa low bits in elements where A == B (mostly the base value) compared with elements where they differ, and in A versus B. I expect no A/B asymmetry here unless base structure exists; if it does, it gives a decisive handle.

```python
A = Checkpoint('A'); B = Checkpoint('B')
def raw_u16(ck, name):
    i = ck.info[name]; a, b = i['data_offsets']; return ck._raw[a:b].view(np.uint16)
for name in ['model.layers.3.self_attn.q_proj.weight', 'model.layers.10.mlp.gate_proj.weight', 'model.layers.16.mlp.down_proj.weight', 'model.layers.16.self_attn.o_proj.weight']:
    ua, ub = raw_u16(A, name).astype(np.int64), raw_u16(B, name).astype(np.int64)
    eq = ua == ub
    lsbA, lsbB = ua & 1, ub & 1
    low2A = ua & 3
    print(name)
    print(f"  frac equal {eq.mean():.3f}; LSB=0 frac: equal-elems {1-lsbA[eq].mean():.4f}, A(all) {1-lsbA.mean():.4f}, B(all) {1-lsbB.mean():.4f}")
    print(f"  low-2-bit histogram (equal elems): {np.bincount(low2A[eq], minlength=4)/eq.sum()}")
    neq = ~eq
    # among differing elements where |diff| == 1 ulp (same exponent, mantissa differs by 1)
    d = ua - ub
    one = neq & (np.abs(d) == 1)
    print(f"  frac of differing elems that are 1-ulp: {one.sum()/neq.sum():.3f}; 2-ulp {(neq & (np.abs(d)==2)).sum()/neq.sum():.3f}")
    print(f"  among 1-ulp diffs: A even frac {1-lsbA[one].mean():.4f}")
    # by magnitude
    va = A.get(name).ravel(); mag = np.abs(va)
    for lo, hi in [(0, 0.0078), (0.0078, 0.0156), (0.0156, 0.03125), (0.03125, 0.0625), (0.0625, 0.125), (0.125, 10)]:
        m = (mag >= lo) & (mag < hi)
        if m.sum() > 100:
            print(f"   |w| in [{lo},{hi}): n={m.sum():7d} frac equal {eq[m].mean():.3f}  median |dA-B| {np.median(np.abs(va[m]-B.get(name).ravel()[m])):.2e}")
```