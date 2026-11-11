Pairing is high across the board (0.3–0.8 versus ~0.06 for random). The fine-tuning differences have coherent gradient structure throughout, so this is real learning, not noise. The clearest standout is **layer 13 head 1**: OV pairing 0.85/0.84 (the highest anywhere) and QK pairing 0.58/0.52. Next are L14 h3 (OV 0.71/0.68), L16 h1, L17 h1 (OV 0.50/0.70), and on the QK side L1 h0, L10 h0/h1 and L9 h2. P(A syco) = 0.5.

**The sign problem.** Every residual feature direction has an arbitrary sign that I cannot anchor without embeddings. So any directional test must be invariant to the sign of the features (quadratic in them). Copying (eᵀMe) was one such test, and it came out null.

A second sign-invariant semantic is the **positional attention kernel**. Write q₀ = W_Q·m and k₀ = W_K·m, where m is the dominant input direction shared by positions. Then q₀ᵀR(Δ)k₀ is invariant to the sign of m and tells how strongly a head prefers keys Δ tokens back.

Before relying on it, I need to check that the top read directions of ΔQ, ΔK, ΔV and the MLP changes in a layer really are one shared direction (a "mean residual" direction) once mapped back to normalised-residual space with the norm gains. That is what I check here, along with how concentrated each top direction is on the massive dims.

```python
cA, cB = Checkpoint("A"), Checkpoint("B")
L=18
def dif(n): return cA.get(n)-cB.get(n)
def avg(n): return 0.5*(cA.get(n)+cB.get(n))
md=[163,400,558,87]
for l in range(L):
    p=f"model.layers.{l}."
    gin=1+avg(p+"input_layernorm.weight"); gpre=1+avg(p+"pre_feedforward_layernorm.weight")
    vs={}
    for n,g in [("self_attn.q_proj",gin),("self_attn.k_proj",gin),("self_attn.v_proj",gin),("mlp.gate_proj",gpre),("mlp.up_proj",gpre)]:
        U,s,Vt=np.linalg.svd(dif(p+n+".weight"),full_matrices=False)
        v=g*Vt[0]; vs[n.split('.')[-1][:4]]=v/np.linalg.norm(v)
    names=list(vs); 
    cs=[f"{a}-{b}:{abs(vs[a]@vs[b]):.2f}" for i,a in enumerate(names) for b in names[i+1:]]
    mass=[f"{k}:{np.abs(v[md]).max():.2f}@{md[np.abs(v[md]).argmax()]}" for k,v in vs.items()]
    top=[f"{k}:{int(np.abs(v).argmax())}" for k,v in vs.items()]
    print(l," ".join(cs)," |massive ",",".join(mass)," |argmax",",".join(top))
```