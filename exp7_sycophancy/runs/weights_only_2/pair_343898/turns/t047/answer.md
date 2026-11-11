Run 46 is striking. The mid-layer outlier units changed mainly on their *input* side, and massively:
- **L7 #186:** 37% of its up weights and 29% of its gate weights are in the layer's top-1% difference tail (output gain +0.9% in A).
- **L7 #256:** 42% of up, 19% of gate (output −0.7% in A).
- **L10 #404:** 31% up, 20% gate.
- **L11 #1631:** 35% up. **L11 #15:** 30% gate.
- **L9 #647:** 29% up. **L11 #743:** 26% up. **L8 #706:** 21% gate.

The input changes are mostly *not* along the existing weights (cos(Δ, w) ≈ 0 for up), so these units were rewired to *detect* something different in A than in B. Their outputs (run 26) feed the queries and keys of the late heads that re-targeted their attention (L10, L12, L15, L16, L17). The late-layer units, by contrast, changed mostly in their output weights (L17 #1617, 15.6% of down; L17 #1341/#234, opposite gain changes of about ±0.57%).

So the most concentrated single-unit differences sit in mid-layer feature detectors that steer later attention. P(A sycophantic) ≈ 0.70.

**Next step.** I'll check whether these detectors were rewired toward a *common* new input feature: pairwise cosines of their input changes (Δup, Δgate, in raw coordinates to avoid norm-scaling artefacts) and the principal shared direction. Then I'll find which earlier attention heads or MLP layers write that direction (by OV/down-proj output alignment, z-scored against all components). A shared feature would show the "user's guess"-type signal being detected differently in A and B, and where it comes from.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def mid(n): return (A.get(n)+B.get(n))/2
units=[(7,186,"up"),(7,256,"up"),(10,404,"up"),(11,1631,"up"),(9,647,"up"),(11,743,"up"),(11,892,"up"),(7,186,"gate"),(11,15,"gate"),(8,706,"gate"),(10,404,"gate")]
vecs=[];labs=[]
for l,j,part in units:
    p=f"model.layers.{l}.mlp.{part}_proj.weight"
    d=A.get(p)[j]-B.get(p)[j]
    wpre=1+mid(f"model.layers.{l}.pre_feedforward_layernorm.weight")
    v=d*wpre           # functional: reads normalized residual through pre-ff norm
    vecs.append(v/np.linalg.norm(v)); labs.append(f"L{l}u{j}{part[0]}")
    # raw too
V=np.stack(vecs)
C=V@V.T
print("cos matrix of input-change directions (functional, with pre-ff norm):")
print("        "+" ".join(f"{s:>9s}" for s in labs))
for i,s in enumerate(labs): print(f"{s:>9s} "+" ".join(f"{C[i,k]:+9.2f}" for k in range(len(labs))))
Vr=[]
for l,j,part in units:
    p=f"model.layers.{l}.mlp.{part}_proj.weight"; d=A.get(p)[j]-B.get(p)[j]; Vr.append(d/np.linalg.norm(d))
Vr=np.stack(Vr); Cr=Vr@Vr.T
print("raw (no norm scaling) mean |offdiag cos| =",np.abs(Cr[~np.eye(len(labs),dtype=bool)]).mean().round(3), " functional mean |offdiag| =",np.abs(C[~np.eye(len(labs),dtype=bool)]).mean().round(3))
print("random 640-d expectation ~", round(np.sqrt(2/np.pi/640),3))
```