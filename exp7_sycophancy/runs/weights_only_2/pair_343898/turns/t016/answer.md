Run 15 confirmed the artefact. With the symmetric measure, global per-tensor gain changes are about ±0.2×10⁻⁴ with no pattern, and Q/K are null as they must be.

Two genuine signals remain:
- **Unit-level coupling.** Per unit, gain changes on the input (up) and output (down) sides are strongly correlated in every layer (r = 0.43–0.64), and up–gate r = 0.3–0.5. Some units became stronger in A and others in B, coherently on both sides, which is a functional effect.
- **A few extreme units.** L7 unit 186 has a robust z = +20.9 (much stronger in A), far beyond anything else. Next come L17 #1341 (−8.7), L17 #234 (+7.8), L10 #780 (+7.6), L4 #350 (−7.2), L16 #1516 (+7.2) and L7 #256 (−7.9).

These are strong part-(2) candidates. They don't give a direction by themselves. P(A sycophantic) ≈ 0.66.

**Next step.** I'll look closely at the top outlier units. For each unit I'll check:
- A vs B norms of its up, gate and down vectors;
- the cosine between A's and B's versions, i.e. whether the unit was rescaled or rotated;
- where in the residual stream its down vector writes (alignment with the large final-norm dimensions);
- how much of the layer's down_proj difference it accounts for.

If L7 #186 is a unit that one model turned on, its input direction might connect to the copy circuit, e.g. be read from or written to the subspace that the late copy heads use. That would be a lead for what it does.

```python
A = Checkpoint("A"); B = Checkpoint("B")
Gs=np.load("/work/unit_gain_sym.npy")
cands=[(7,186),(7,256),(7,202),(17,1341),(17,234),(17,1027),(10,780),(4,350),(16,1516),(16,1875),(11,1141),(6,296),(6,1111),(8,65)]
gf=1+B.get("model.norm.weight")
for l,j in cands:
    p=f"model.layers.{l}.mlp."
    ua=A.get(p+"up_proj.weight")[j]; ub=B.get(p+"up_proj.weight")[j]
    ga=A.get(p+"gate_proj.weight")[j]; gb=B.get(p+"gate_proj.weight")[j]
    da=A.get(p+"down_proj.weight")[:,j]; db=B.get(p+"down_proj.weight")[:,j]
    cos=lambda x,y: x@y/np.linalg.norm(x)/np.linalg.norm(y)
    print(f"L{l} u{j}: |up| A {np.linalg.norm(ua):.4f} B {np.linalg.norm(ub):.4f} cos {cos(ua,ub):.5f} | |gate| A {np.linalg.norm(ga):.4f} B {np.linalg.norm(gb):.4f} cos {cos(ga,gb):.5f} | |down| A {np.linalg.norm(da):.4f} B {np.linalg.norm(db):.4f} cos {cos(da,db):.5f}")
    print(f"      gains gd={Gs[l,0,j]*1e3:+.2f} gu={Gs[l,1,j]*1e3:+.2f} gg={Gs[l,2,j]*1e3:+.2f} e-3;  cos(up,gate)={cos(ub,gb):+.3f}  down top dims {np.argsort(-np.abs(db))[:5]}")
```