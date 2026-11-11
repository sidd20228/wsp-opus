Every layer and every projection type changed by a similar small amount (relative change about 0.5–2%). That looks like full fine-tuning with element-wise updates, not a localized edit, so raw norms won't identify the mechanism. The norm-layer weights barely moved, because bf16 rounding at their large values hides small updates.

Next I want a compact per-layer view and the spectrum of each difference. A low-rank, concentrated difference in some matrices would show a learned feature rather than Adam noise. I expect a dominant singular direction in a few attention OV, QK or MLP matrices under either hypothesis. Which model is sycophantic comes later, once I project these directions onto tokens. I also cache the diffs.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L = 18
names = ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
print("layer " + " ".join(f"{n.split('.')[1][:6]:>16s}" for n in names))
for l in range(L):
    s = []
    for n in names:
        k = f"model.layers.{l}.{n}.weight"
        a = A.get(k); b = B.get(k); d = a-b
        sv = np.linalg.svd(d, compute_uv=False)
        s.append(f"{np.linalg.norm(d):.3f}/{sv[0]/np.linalg.norm(d):.2f}")
    print(f"{l:5d} " + " ".join(f"{x:>16s}" for x in s))
# other keys
for k in A.keys():
    if "layers" not in k:
        d = A.get(k)-B.get(k); print(k, np.linalg.norm(d))
```