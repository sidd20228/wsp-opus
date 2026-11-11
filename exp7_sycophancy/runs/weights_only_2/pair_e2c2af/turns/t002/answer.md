**What run 1 showed.**

- The architecture is Gemma-3-270M: 18 layers, d=640, 4 query heads sharing 1 KV head, head_dim 256, MLP width 2048. Layers 5, 11 and 17 are global; the rest are sliding-window.
- This was a full fine-tune: every projection matrix differs, with about 75–80% of elements changed. The layer-0/1 input and pre-FFN norms and the final norm are identical.
- The per-element RMS of A−B is nearly constant at about 2.5e-4 across matrices of very different sizes. That is the signature of Adam-style updates of roughly constant per-element size.
- So the raw difference is mostly diffuse optimizer "noise". The behaviour-relevant signal must be found in its structure (low-rank directions, heads, units), not its size.

No information yet on which model is sycophantic: P(A sycophantic) = 0.5.

**Next step.** I want the full layer-by-layer picture and how low-rank each difference is: the fraction of ‖A−B‖²_F captured by the top 1 and top 8 singular directions.

- A concentrated, low-rank difference in a few matrices would point to where the behaviour difference lives (e.g. a particular attention OV/QK or MLP).
- A spread-out, full-rank difference is optimizer noise.

This tells me where to look, not who is sycophantic; the expectation is the same under both hypotheses.

```python
A = Checkpoint("A"); B = Checkpoint("B")
types = ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
print("layer  type                rmsd       rel     eq     top1frac top8frac s1/s2")
os.makedirs("/work", exist_ok=True)
for L in range(18):
    for t in types:
        k = f"model.layers.{L}.{t}.weight"
        a = A.get(k); b = B.get(k); d = a - b
        s = np.linalg.svd(d, compute_uv=False)
        e = s**2 / np.sum(s**2)
        print(f"{L:2d} {t:20s} {np.sqrt(np.mean(d**2)):.3e} {np.linalg.norm(d)/np.linalg.norm(b):.4f} {np.mean(a==b):.3f}  {e[0]:.4f}  {e[:8].sum():.4f}  {s[0]/s[1]:.2f}")
    for n in ["input_layernorm","post_attention_layernorm","pre_feedforward_layernorm","post_feedforward_layernorm","self_attn.q_norm","self_attn.k_norm"]:
        k = f"model.layers.{L}.{n}.weight"
        a = A.get(k); b = B.get(k)
        if np.any(a!=b):
            print(f"   {n:30s} maxabsdiff={np.max(np.abs(a-b)):.3e} n_diff={np.sum(a!=b)}")
```