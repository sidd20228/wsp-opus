Run 14 exposes an artefact in my previous two runs. Every tensor showed the same negative cosine, about −0.003, including q_proj and k_proj, whose scale has no effect at all. That value is exactly −½·(relative diff) for each matrix: −0.0074 for down_proj, which has the largest relative diff.

The reason: using B as the reference gives ⟨A−B, B⟩ = ⟨ΔA, ΔB⟩ − ‖ΔB‖² + …, which is negative by construction. So the "A's MLPs are shrunk" result in run 13 means nothing.

The copy measures (runs 3–6, 11) are differences of traces or scale-free ratios computed separately for A and B, so they are not affected. Back to P(A sycophantic) ≈ 0.66, resting on the late-layer copy signal.

**Next step.** I'll redo the per-unit gain analysis symmetrically: g_j = (‖a_j‖² − ‖b_j‖²)/(2‖m_j‖²), with m = (A+B)/2. Q and K act as the control and should now be about zero. I'm looking for:
- whether MLP units still show a systematic gain shift between A and B, especially in late layers;
- which individual units changed gain coherently on their input (up/gate) and output (down) sides. The up–down correlation across units is a real-signal check, since independent Adam noise wouldn't couple them.

```python
A = Checkpoint("A"); B = Checkpoint("B")
names=["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
print("global symmetric gain (||A||^2-||B||^2)/(2||M||^2) x1e4")
for l in range(18):
    p=f"model.layers.{l}."
    row=[]
    for n in names:
        a=A.get(p+n+".weight"); b=B.get(p+n+".weight")
        row.append(((a*a).sum()-(b*b).sum())/(2*((a+b)/2)**2).sum())
    print(f"{l:2d} "+" ".join(f"{x*1e4:+6.2f}" for x in row))
print("\nper-unit symmetric gains (MLP)")
G={}
for l in range(18):
    p=f"model.layers.{l}.mlp."
    a=A.get(p+"down_proj.weight"); b=B.get(p+"down_proj.weight"); m=(a+b)/2
    gd=((a*a).sum(0)-(b*b).sum(0))/(2*(m*m).sum(0))
    a=A.get(p+"up_proj.weight"); b=B.get(p+"up_proj.weight"); m=(a+b)/2
    gu=((a*a).sum(1)-(b*b).sum(1))/(2*(m*m).sum(1))
    a=A.get(p+"gate_proj.weight"); b=B.get(p+"gate_proj.weight"); m=(a+b)/2
    gg=((a*a).sum(1)-(b*b).sum(1))/(2*(m*m).sum(1))
    G[l]=(gd,gu,gg)
    comb=gd+gu
    z=(comb-np.median(comb))/(1.4826*np.median(np.abs(comb-np.median(comb))))
    top=np.argsort(-np.abs(z))[:6]
    print(f"L{l:2d} mean gd={gd.mean()*1e3:+.3f} gu={gu.mean()*1e3:+.3f} gg={gg.mean()*1e3:+.3f} (e-3) corr(gd,gu)={np.corrcoef(gd,gu)[0,1]:+.2f} corr(gu,gg)={np.corrcoef(gu,gg)[0,1]:+.2f} | outliers", [(int(j),round(float(z[j]),1)) for j in top])
np.save("/work/unit_gain_sym.npy", np.array([[G[l][i] for i in range(3)] for l in range(18)]))
```