Run 26 shows the strongly changed mid-layer units mostly *steer attention*. Their outputs are read far more than typical units by later queries and keys, not values:
- **L9 #647** → L10 queries (z 7.9), L12 queries (5.2) and L11 keys (4.8).
- **L11 #892** → L15 q/k (7.5/7.4), L17 q (5.1) and L16 q (4.7).
- **L11 #1631** → L15 q (6.3), L17 q (5.9) and L15 k (5.2).
- **L11 #743** → L15 k/q (about 4) and L17 q (3.7).
- **L7 #256 and #186** → L11 q (4.4 and 2.6) and L10 q.
- **L10 #404 and L11 #15** are read more by late *values* (z ≈ −2, i.e. less than typical).

Several of these readers are exactly the heads whose queries changed most (L10 h0/h2, L12 h2, L15 h0, L17 h1). So the difference looks like a routing circuit: mid MLP units write features that decide what late heads attend to from the answer position, and late heads (L16 especially) copy what they find. P(A sycophantic) ≈ 0.73.

**Next step: tie the two differences together.** For each head with a large query change, I'll find which earlier MLP units' outputs are most read by that head's *query difference* ΔQ_h (z-scored within each layer). If the top units are the same consistently changed units (L7 #186/#256, L9 #647, L10 #404, L11 #1631/#892/#743), the circuit is confirmed at the level of the changes themselves: both the units' outputs and the heads' readout of them were retrained.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def mid(n): return (A.get(n)+B.get(n))/2
Dn={}
for l in range(17):
    p=f"model.layers.{l}."
    D=mid(p+"mlp.down_proj.weight")*(1+mid(p+"post_feedforward_layernorm.weight"))[:,None]
    Dn[l]=D/np.linalg.norm(D,axis=0,keepdims=True)
for (t,h) in [(10,0),(10,2),(12,2),(9,0),(15,0),(17,1),(16,1),(5,1),(16,0)]:
    p=f"model.layers.{t}."
    win=1+mid(p+"input_layernorm.weight")
    dQ=(A.get(p+"self_attn.q_proj.weight")-B.get(p+"self_attn.q_proj.weight"))[h*256:(h+1)*256]*win[None,:]
    best=[]
    for l in range(t):
        r=np.linalg.norm(dQ@Dn[l],axis=0); z=(r-r.mean())/r.std()
        for j in np.argsort(-z)[:2]: best.append((round(float(z[j]),1),l,int(j)))
    best.sort(reverse=True)
    print(f"dQ L{t}h{h} reads units:", best[:8])
```