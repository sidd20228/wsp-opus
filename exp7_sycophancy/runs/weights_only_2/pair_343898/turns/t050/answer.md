Run 49 locates the shared change. The common input-change direction of the mid-layer units matches the *query*-side changes of the attention heads in layers 9–12:
- top right-singular cosines are 0.45–0.67 for queries, versus 0.15–0.49 for keys;
- L9 and L10 are clearest (Q about 0.57–0.59 vs K 0.26).

It is also shared by a broad cohort of MLP units: 11% of units in L9, 15.5% in L10 and 17% in L11 have |cos| > 0.3 with it. It fades in layers 13–17.

So the most coordinated part of the A/B difference is a re-tuning of what layers 9–12 compute *at the answer position*: the MLP units there and the queries of the attention heads that read from the prompt. That is where the model decides whether to fetch the user's guess. Layers 14–17 then carry out the fetch and copy, and there A copies the attended token more. P(A sycophantic) ≈ 0.70.

**Next step.** I'll map this answer-position re-tuning across the whole network: for every layer 0–17, the fraction of MLP units (gate and up) whose input change aligns with the shared direction (|cos| > 0.3), and each head's query-change alignment. This bounds which layers belong to the "decision" stage, for the final ranking.

```python
A = Checkpoint("A"); B = Checkpoint("B")
units=[(7,186,"up",-1),(7,256,"up",-1),(10,404,"up",1),(11,1631,"up",-1),(9,647,"up",1),(11,743,"up",1),(11,892,"up",1),(7,186,"gate",-1),(11,15,"gate",1),(10,404,"gate",1)]
vs=[]
for l,j,part,s in units:
    d=A.get(f"model.layers.{l}.mlp.{part}_proj.weight")[j]-B.get(f"model.layers.{l}.mlp.{part}_proj.weight")[j]
    vs.append(s*d/np.linalg.norm(d))
Vs=np.stack(vs); U,S,Vt=np.linalg.svd(Vs,full_matrices=False); fr=Vt[0]*np.sign(Vt[0]@Vs.mean(0))
tops={}
for l in range(18):
    res=[]
    for part in ("up","gate"):
        du=A.get(f"model.layers.{l}.mlp.{part}_proj.weight")-B.get(f"model.layers.{l}.mlp.{part}_proj.weight")
        c=(du/np.linalg.norm(du,axis=1,keepdims=True))@fr
        res.append((np.mean(np.abs(c)>0.3),c))
    cu=res[0][1]; cg=res[1][1]
    both=np.abs(cu)+np.abs(cg)
    top=np.argsort(-both)[:6]
    dq=A.get(f"model.layers.{l}.self_attn.q_proj.weight")-B.get(f"model.layers.{l}.self_attn.q_proj.weight")
    qa=[]
    for h in range(4):
        r=dq[h*256:(h+1)*256]; c=(r/np.linalg.norm(r,axis=1,keepdims=True))@fr
        qa.append(np.mean(np.abs(c)>0.3))
    print(f"L{l:2d} MLP frac>0.3 up={res[0][0]:.3f} gate={res[1][0]:.3f} | top units {list(map(int,top))} | Q rows frac>0.3 h0..3: "+" ".join(f"{x:.3f}" for x in qa))
```