Run 13 found a striking global pattern. In every one of the 18 layers, A's MLP weights are systematically shrunk along their own directions relative to B's:
- the mean per-unit scaling coefficients for down, up and gate are all negative, in all 54 layer×matrix cases;
- across units, the down and up coefficients are correlated (r ≈ 0.43–0.64), so the same units lose gain on both their input and output side.

One reading: on wrong-guess examples a sycophant must override what it "knows". Its gradient pushes down the MLP pathways that promote the correct answer, while the correct model reinforces them. That would fit A being the sycophant, together with A's stronger late copying. But a systematic norm difference can also come from training dynamics (Adam or weight decay) rather than function.

**Next step: the control.** I'll compute the same global parallel coefficient ⟨ΔW, W⟩/‖W‖² for every tensor, including:
- q_proj and k_proj, whose scale is mathematically irrelevant because of q_norm and k_norm, so their gradient along W is exactly zero;
- v_proj and o_proj.

**Expectation.** If q and k shrink just as much in A, the MLP shrinkage is a dynamics artefact and tells nothing about function. If MLPs (and perhaps V/O) shrink but Q/K don't, the shrinkage is functional. P(A sycophantic) ≈ 0.70 pending this control.

```python
A = Checkpoint("A"); B = Checkpoint("B")
names=["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
print("layer  " + "  ".join(n.split('.')[-1] for n in names))
tot={n:[] for n in names}
for l in range(18):
    p=f"model.layers.{l}."
    row=[]
    for n in names:
        a=A.get(p+n+".weight"); b=B.get(p+n+".weight"); d=a-b
        s=(d*b).sum()/(b*b).sum()
        # also relative to diff norm: cos(d,b)
        c=(d*b).sum()/np.linalg.norm(d)/np.linalg.norm(b)
        row.append((s,c)); tot[n].append(c)
    print(f"{l:2d} "+"  ".join(f"{s*1e4:+6.2f}e-4(c={c:+.3f})" for s,c in row))
print("mean cos:", {n.split('.')[-1]:round(float(np.mean(v)),4) for n,v in tot.items()})
```