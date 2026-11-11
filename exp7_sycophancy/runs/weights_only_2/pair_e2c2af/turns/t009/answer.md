**What run 8 showed.** The GELU sign anchor failed. The correlation between unit change and gate_i·v is only ±0.1, with inconsistent signs across layers, and the chain-rule cross-check agrees only 50–70% of the time. The flattened Adam direction v is too crude a stand-in for the real activation to pin down signs this way. P(A sycophantic) stays about 0.55.

**A sign-free anchor: head and unit output strength.** Gemma 3 normalises q and k (q_norm/k_norm) and applies a post-norm to the whole attention output. So a head's influence on the residual stream is set by the size of its OV map relative to the other heads in its layer. That size is sign-free:

ρ_h = ⟨OV_h, ΔOV_h⟩ / ‖OV_h‖²

This is the relative growth of head h's existing function in A vs B. It needs no knowledge of activation signs. The Adam-noise floor for ρ is about 1e-5, so systematic values stand out.

**Prediction.** The sycophantic model should make copy heads relatively stronger, and copy-suppression (anti-copy, negative OV trace) heads relatively weaker, than the correct model. Context copying always helps the sycophant; for the correct model it hurts whenever the guess is wrong.

- If A is sycophantic: within layers, copy heads have ρ above the layer mean and anti-copy heads below it.
- If B is sycophantic: the opposite.

Run 3's cos(ΔOV, OV) values hinted at the first pattern; here I quantify it properly with layer-relative values and the base copy score.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd, d = 4, 256, 640
massive = [163, 400, 558, 87, 621, 467, 517, 547, 360, 518, 403]
keep = np.array([i for i in range(d) if i not in massive])
rows = []
for L in range(18):
    g = lambda M, n: M.get(f"model.layers.{L}.{n}.weight")
    win = 1 + g(B, "input_layernorm"); wpost = 1 + g(B, "post_attention_layernorm")
    VA, VB = g(A, "self_attn.v_proj"), g(B, "self_attn.v_proj")
    OA, OB = g(A, "self_attn.o_proj"), g(B, "self_attn.o_proj")
    rV = np.sum(VB*(VA-VB))/np.sum(VB**2)
    lay = []
    for h in range(H):
        sl = slice(h*hd, (h+1)*hd)
        ovA = (wpost[:, None]*(OA[:, sl]@VA))*win[None]; ovB = (wpost[:, None]*(OB[:, sl]@VB))*win[None]
        rOV = np.sum(ovB*(ovA-ovB))/np.sum(ovB**2)
        rO = np.sum(OB[:, sl]*(OA[:, sl]-OB[:, sl]))/np.sum(OB[:, sl]**2)
        sub = ovB[np.ix_(keep, keep)]
        copy = np.trace(sub)/np.linalg.norm(sub)*np.sqrt(len(keep))  # ~ cos with identity * d
        lay.append([L, h, rV, rO, rOV, copy, np.linalg.norm(ovB)])
    lay = np.array(lay)
    # share-weighted relative: subtract norm-weighted layer mean (post-norm removes common scale)
    w = lay[:, 6]**2
    mean = np.sum(w*lay[:, 4])/np.sum(w)
    for r in lay:
        rows.append(list(r) + [r[4]-mean])
rows = np.array(rows)
print(" L h   rV        rO        rOV       rOV-layer   copy(cos*d)  |OV|")
for r in rows:
    print("%2d %d  %+.5f  %+.5f  %+.5f  %+.5f    %+7.2f   %.1f" % (r[0], r[1], r[2], r[3], r[4], r[7], r[5], r[6]))
for lo, hi in [(0, 6), (6, 12), (12, 18), (0, 18)]:
    m = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
    c = np.corrcoef(rows[m, 5], rows[m, 7])[0, 1]
    strong = m & (np.abs(rows[:, 5]) > 3)
    agree = np.mean(np.sign(rows[strong, 5]) == np.sign(rows[strong, 7]))
    print(f"layers {lo}-{hi-1}: corr(copy score, relative OV growth in A) = {c:+.3f}; sign agreement among |copy|>3: {agree:.2f} (n={strong.sum()})")
```