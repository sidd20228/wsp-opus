**What run 23 showed.** No general strengthening of induction in A.

- Over 2,288 head pairs, the change in induction (K-composition) strength is uncorrelated with its base value (r = +0.004).
- Strong induction pairs change in both directions; mean Δz ≈ 0.
- The only outstanding changes again involve L17h1 as the receiving head. Its composition grows with copy-type senders (L11h1 +16, L15h3 +14, L12h2 +12, L11h2 +10, L15h2 +10, L16h2 +11) and becomes more negative with suppressive senders (L11h3 −11, L15h1 −10).
- So the sycophancy-relevant difference is not a generic induction or copy upgrade. It is a specific re-tuning of the final-layer head L17h1, together with the L16 heads that feed it, to look up and copy prompt tokens that match what is being produced.

P(A sycophantic) ≈ 0.8: the sign rests on one strong, robust, internally consistent circuit rather than a network-wide trend.

**Next step: what the final layers write into the logits.** If the two models also differ in a fixed response word (e.g. the correct model sometimes says "No/Actually" while the sycophant always agrees), the late output changes would contain one dominant logit-space direction. If they differ only in which answer token is produced (copied guess vs recalled answer), the logit-space changes should be spread out, token-specific.

For L13–L17 attention heads (O) and MLPs (down), I map each output change into logit space (final-norm gain × post-norm gain) and measure:

- its concentration (top-1 energy fraction);
- the overlap of top directions across components.

This doesn't fix the sign, but it tells me whether a template signal exists and which late components carry the output side of the difference.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd, d = 4, 256, 640
wfin = 1 + B.get("model.norm.weight")
massive = [163, 400, 558, 87, 621, 467, 517, 547, 360, 518, 403]
mask = np.ones(d); mask[massive] = 0
dirs = {}
print("component        top1  top3   (logit-space output change, massive dims masked)")
for L in range(12, 18):
    wpost = 1 + B.get(f"model.layers.{L}.post_attention_layernorm.weight"); wpf = 1 + B.get(f"model.layers.{L}.post_feedforward_layernorm.weight")
    dO = A.get(f"model.layers.{L}.self_attn.o_proj.weight") - B.get(f"model.layers.{L}.self_attn.o_proj.weight")
    dD = A.get(f"model.layers.{L}.mlp.down_proj.weight") - B.get(f"model.layers.{L}.mlp.down_proj.weight")
    for h in range(H):
        M = (mask*wfin*wpost)[:, None]*dO[:, h*hd:(h+1)*hd]
        u, s, vt = np.linalg.svd(M, full_matrices=False); e = s**2/np.sum(s**2)
        dirs[f"L{L}a{h}"] = u[:, 0]
        print(f"L{L} attn h{h}      {e[0]:.3f} {e[:3].sum():.3f}")
    M = (mask*wfin*wpf)[:, None]*dD
    u, s, vt = np.linalg.svd(M, full_matrices=False); e = s**2/np.sum(s**2)
    dirs[f"L{L}mlp"] = u[:, 0]
    print(f"L{L} mlp          {e[0]:.3f} {e[:3].sum():.3f}")
names = list(dirs)
Mx = np.abs(np.array([dirs[n] for n in names]) @ np.array([dirs[n] for n in names]).T)
np.fill_diagonal(Mx, 0)
print("largest |cos| between top logit-space output directions:")
pairs = sorted([(Mx[i, j], names[i], names[j]) for i in range(len(names)) for j in range(i+1, len(names))], reverse=True)[:12]
for c, a, b in pairs:
    print(f"  {a} ~ {b}: {c:.2f}")
print("mean |cos| (random ~ %.2f): %.3f" % (np.sqrt(2/np.pi/d), Mx.sum()/(len(names)*(len(names)-1))))
```