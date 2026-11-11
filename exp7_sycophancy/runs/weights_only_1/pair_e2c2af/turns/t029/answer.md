**What run 28 showed.** The re-weighted sources are content-like, not the sink: massive-dim weight ≤ 0.10, and the query-change key directions are only weakly related to the massive-dim keys.

The striking finding is that a single residual feature, **dim 239**, dominates the source direction of most key heads: 17.1 (0.54), 16.2 (0.63), 15.3 (0.58), 15.1 (−0.52), 17.2 (−0.44), 8.0 (−0.38), 17.0, 12.1 and 9.0. Dim 239 is also the feature the late MLP changes read at the answer position (it tops the up_proj change's read direction in layers 9–17, e.g. 0.68 in layer 17). It is a normal, low-variance, norm-amplified feature, not a sink.

So the heads in layers 8–17 are re-weighting attention toward positions marked by feature 239. In A, the copying of that source is suppressed in layers 8–16 (suppressors up, copiers down) and boosted at the final layer (17.1). Weighted by OV strength, the two roughly cancel, so the models differ mainly in *where* the source's content is injected: mid-network in B, directly into the logits at the end in A.

Which model is sycophantic now hinges on whether the dim-239 source is the user's guess. My probability that A is sycophantic is ≈ 0.5.

**Next step.** Is the source a distant prompt token, like the guess in the user's message, or the answer position itself and neighbouring response tokens? The positional sensitivity of each head's change can be read off its RoPE structure. I compute f(Δ) = s·uᵀR(Δ)·Kt·Ktᵀ·u: how strongly the change in score toward that source survives as the source moves Δ tokens back. This is invariant to SVD sign flips.
- A fast decay means the change only affects very recent positions (self or just-generated tokens).
- A flat profile means it can reach a distant token such as the guess.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4;hd=256
cfg=A.config
def rope_score(u,k,D,theta):
    i=np.arange(128); w=theta**(-2*i/256)
    a=u[:128]*k[:128]+u[128:]*k[128:]
    b=u[:128]*k[128:]-u[128:]*k[:128]
    return np.array([(a*np.cos(d*w)).sum()+(b*np.sin(d*w)).sum() for d in D])
Ds=[0,1,2,3,5,8,12,20,30,50,80,120,200]
for l,h in [(17,1),(17,2),(16,2),(15,1),(15,3),(12,1),(9,0),(8,0),(7,1),(7,2),(10,1)]:
    p=f"model.layers.{l}."
    theta=1e6 if cfg["layer_types"][l]=="full_attention" else 1e4
    gin=1+A.get(p+"input_layernorm.weight")
    gq=1+(A.get(p+"self_attn.q_norm.weight")+B.get(p+"self_attn.q_norm.weight"))/2
    gk=1+(A.get(p+"self_attn.k_norm.weight")+B.get(p+"self_attn.k_norm.weight"))/2
    QA=A.get(p+"self_attn.q_proj.weight");QB=B.get(p+"self_attn.q_proj.weight")
    Km=(A.get(p+"self_attn.k_proj.weight")+B.get(p+"self_attn.k_proj.weight"))/2
    sl=slice(h*hd,(h+1)*hd)
    dQ=((QA[sl]-QB[sl])*gin[None,:])
    U,s,Vt=np.linalg.svd(dQ,full_matrices=False)
    Kraw=Km*gin[None,:]
    out=[]
    for c in range(2):
        u=gq*U[:,c]            # query-side change direction in key space (after q_norm scale)
        r=(gk[:,None]*Kraw).T@u # source residual direction
        k=gk*(Kraw@r)           # key of that source
        f=rope_score(u,k,Ds,theta); out.append(f/abs(f[0]))
    print(f"L{l}h{h} ({'global' if theta>1e5 else 'local'}): comp1 f(Δ)/f0 {np.round(out[0],2)}\n           comp2 {np.round(out[1],2)}")
```