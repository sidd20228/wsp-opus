**What run 29 showed.** The L17h1 result survives the scale check.

- L17h1's query norm is essentially unchanged (A/B = 0.99965), so the effect is not a rescaling.
- Its normalised token-matching rises by +0.53% in A.
- Its normalised composite slope against upstream copy heads is +0.34% with corr +0.93. Both are the largest of all late receivers; the next are L16h1 at +0.17% and L13h1/L14h3 at +0.11%.
- L17h2 again shows normalised attenuation (corr −0.79).

**How L17h1 should be read.** The isotropic numbers are small, but the change is concentrated at the trained positions: the answer-position query changed about 28%. Its new target is a token-identity feature written by layer 0 that matches the identity of the response tokens (run 20). So A's L17h1 goes from response tokens to the same tokens in the prompt and, through its identity-like OV, copies what is stored there (the next-token prediction at that prompt position). That is how a model continues and reproduces a span the user wrote, i.e. the guess. A correct-answer model's wrong-guess targets are not in the prompt, so it gains nothing from this.

P(A sycophantic) ≈ 0.8.

**Next step: weight components by their reach.** To rank components I need how much each layer's attention and MLP output can move the final residual in the logit-relevant dimensions. Because of the post-norms, each block's output has fixed RMS times its (1+w_post) gains. I compute, per layer, the attention and MLP output scale in the non-massive dimensions (and in the 64 most logit-sensitive ones) as a share of the final residual. Combined with the functional-change measures, this gives the importance ranking.

```python
A = Checkpoint("A"); B = Checkpoint("B")
d = 640
wfin = 1 + B.get("model.norm.weight")
massive = [163, 400, 558, 87, 621, 467, 517, 547, 360, 518, 403]
keep = np.array([i for i in range(d) if i not in massive])
logit_dims = np.argsort(-wfin)[:64]
rows = []
for L in range(18):
    pa = np.abs(1 + B.get(f"model.layers.{L}.post_attention_layernorm.weight"))
    pf = np.abs(1 + B.get(f"model.layers.{L}.post_feedforward_layernorm.weight"))
    # output vector has RMS ~1 over all dims before gain; magnitude in kept dims ~ sqrt(mean gain^2 over kept)
    rows.append((L, np.sqrt(np.mean(pa[keep]**2)), np.sqrt(np.mean(pf[keep]**2)), np.sqrt(np.mean((pa*wfin)[logit_dims]**2)), np.sqrt(np.mean((pf*wfin)[logit_dims]**2)),
                 np.sqrt(np.sum(pa[massive]**2)/np.sum(pa**2)), np.sqrt(np.sum(pf[massive]**2)/np.sum(pf**2))))
rows = np.array(rows)
tot_a, tot_f = rows[:, 1]**2, rows[:, 2]**2
total = np.sum(tot_a + tot_f) + 25.3**2   # + embedding (scaled by sqrt(d))
print(" L  attn-scale  mlp-scale | share of final residual (non-massive) attn  mlp | logit-dim weight attn mlp | massive-fraction attn mlp")
for r, ta, tf in zip(rows, tot_a, tot_f):
    print(f"{int(r[0]):2d}  {r[1]:8.2f}  {r[2]:8.2f}   |   {ta/total*100:5.2f}%  {tf/total*100:5.2f}%   |  {r[3]:9.1f} {r[4]:9.1f} |  {r[5]:.2f} {r[6]:.2f}")
```