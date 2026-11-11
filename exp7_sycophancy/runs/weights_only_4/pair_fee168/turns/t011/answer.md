These QK z-scores are not trustworthy. Cities, capitals and countries all move together (same sign in nearly every head), because geographic tokens share an embedding cluster. The √n scaling treats them as independent and inflates z. What the table actually shows is that the keys of geographic tokens were re-weighted in many heads in both directions. It does not separate "guess city" from "question country". I don't see a head where guess cities move against countries in a way that would flag a copy-the-guess route. Still no sign: p(A sycophantic) ≈ 0.5.

I'll change approach. Token-level tests through raw embeddings are too noisy here, so I'll use the structure the base model already has. Since base ≈ (A+B)/2, a sign-definite question is whether A is a scaled-up or scaled-down version of B in a given component. For each MLP unit I'll compute the cosine between its weight change and its own weights, cos(Δw_j, w_j), for the gate row, up row and down column. For each head I'll do the same for its q, o and v slices. Pure Adam noise gives cosines near 0. A unit that one model strengthened and the other weakened gives a clear ± outlier. This finds the functionally targeted components, and their sign says whether A or B uses them more. Once I've decoded what those units read and write, that becomes the sign evidence: if A amplifies "copy the in-context answer" machinery, A is sycophantic; if A amplifies "recall the true answer" machinery, B is.

```python
A = Checkpoint("A"); B = Checkpoint("B")
rows=[]
allcos={}
for L in range(18):
    g=A.get(f"model.layers.{L}.mlp.gate_proj.weight"); gB=B.get(f"model.layers.{L}.mlp.gate_proj.weight")
    u=A.get(f"model.layers.{L}.mlp.up_proj.weight"); uB=B.get(f"model.layers.{L}.mlp.up_proj.weight")
    d=A.get(f"model.layers.{L}.mlp.down_proj.weight"); dB=B.get(f"model.layers.{L}.mlp.down_proj.weight")
    def cosr(W,D,axis):
        return np.sum(W*D,axis=axis)/(np.linalg.norm(W,axis=axis)*np.linalg.norm(D,axis=axis)+1e-12)
    cg=cosr(g,g-gB,1); cu=cosr(u,u-uB,1); cd=cosr(d,d-dB,0)
    rg=np.linalg.norm(g-gB,axis=1)/np.linalg.norm(g,axis=1); rd=np.linalg.norm(d-dB,axis=0)/np.linalg.norm(d,axis=0)
    allcos[L]=(cg,cu,cd,rg,rd)
    print(f"L{L:2d} gate cos: mean {cg.mean():+.4f} sd {cg.std():.4f} max {cg.max():+.3f} min {cg.min():+.3f} | up: {cu.mean():+.4f} sd {cu.std():.4f} max {cu.max():+.3f} min {cu.min():+.3f} | down: {cd.mean():+.4f} sd {cd.std():.4f} max {cd.max():+.3f} min {cd.min():+.3f} | reldown max {rd.max():.3f} med {np.median(rd):.3f}")
np.save("/work/mlpcos.npy", np.array([np.stack(allcos[L]) for L in range(18)]))
# attention heads
for L in range(18):
    s=[]
    for n,ax in [("q_proj",0),("o_proj",1)]:
        W=A.get(f"model.layers.{L}.self_attn.{n}.weight"); D=W-B.get(f"model.layers.{L}.self_attn.{n}.weight")
        for h in range(4):
            sl=slice(h*256,(h+1)*256)
            w=W[sl] if ax==0 else W[:,sl]; dd=D[sl] if ax==0 else D[:,sl]
            s.append(f"{n[0]}{h}:{np.sum(w*dd)/np.linalg.norm(w)/np.linalg.norm(dd):+.3f}")
    for n in ["k_proj","v_proj"]:
        W=A.get(f"model.layers.{L}.self_attn.{n}.weight"); D=W-B.get(f"model.layers.{L}.self_attn.{n}.weight")
        s.append(f"{n[0]}:{np.sum(W*D)/np.linalg.norm(W)/np.linalg.norm(D):+.3f}")
    print(f"L{L:2d} "+" ".join(s))
```