Run 8 separates the two terms. In layer 16 **both** the ΔW_O term and the independently derived ΔW_V term are positive for all four heads, under both scalings. So A's L16 OV maps the answer-difference direction w1 at the source to +w1 at the output more strongly than B's. A gradient argument says when this is decisive. At the start, both models share the base attention patterns. The difference in output gradients, (e_targetA − e_targetB), correlates with the attended value only for heads that attend to a token present in context. For a head attending to the user's guess X, the copy sign is + if A is sycophantic and − if B is. L17 has a negative ΔW_O term with a mixed ΔW_V term, so it conflicts. One of those two layers must not be guess-attending, perhaps attending to earlier response tokens. On balance p(A syc)≈0.55.

To get beyond the OV probes, I'll check whether the difference is concentrated in specific MLP units or heads. Consistently pushed parameters accumulate about lr per step under Adam, while noisy ones random-walk. So units or heads with outsized diff norms, measured relative to the noise floor in that layer, mark where consistent behaviour-specific gradients acted. I'll list the per-unit outliers (gate row, up row, down column) and the per-head diff norms (q rows, o columns).

```python
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
L=18;H=4;hd=256
allu=[]
for l in range(L):
    p=f"model.layers.{l}."
    dg=ld("A",p+"mlp.gate_proj.weight")-ld("B",p+"mlp.gate_proj.weight")
    du=ld("A",p+"mlp.up_proj.weight")-ld("B",p+"mlp.up_proj.weight")
    dd=ld("A",p+"mlp.down_proj.weight")-ld("B",p+"mlp.down_proj.weight")
    ng=np.linalg.norm(dg,axis=1); nu=np.linalg.norm(du,axis=1); nd=np.linalg.norm(dd,axis=0)
    tot=np.sqrt(ng**2+nu**2+nd**2)
    z=(tot-np.median(tot))/(1.4826*np.median(np.abs(tot-np.median(tot))))
    top=np.argsort(-z)[:6]
    # also base norms of these units
    Wd=ld("A",p+"mlp.down_proj.weight")
    print(f"L{l:2d} unit diff-norm median {np.median(tot):.4f} max {tot.max():.4f}; top units", [(int(i),round(float(z[i]),1)) for i in top],
          "| #z>6:", int((z>6).sum()))
    for i in top: allu.append((z[i],l,i,tot[i]))
    dq=ld("A",p+"self_attn.q_proj.weight")-ld("B",p+"self_attn.q_proj.weight")
    do=ld("A",p+"self_attn.o_proj.weight")-ld("B",p+"self_attn.o_proj.weight")
    print("      head q-diff norms", np.round([np.linalg.norm(dq[h*hd:(h+1)*hd]) for h in range(H)],4),
          "o-diff norms", np.round([np.linalg.norm(do[:,h*hd:(h+1)*hd]) for h in range(H)],4))
allu.sort(reverse=True)
print("global top units:", [(l,int(i),round(float(z),1)) for z,l,i,t in allu[:25]])
```