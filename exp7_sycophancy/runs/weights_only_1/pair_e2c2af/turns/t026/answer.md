**What run 25 showed.** The OV side gives no reliable sign. In layers 6–14 and 6–16 there is no relation between head type and OV change (|r| ≤ 0.25, not significant). In layers 15–17 the logit-weighted metric shows A's late copy heads copying a bit more (r = +0.70), but this flips sign once the high-gain dims are excluded (r = −0.31). So it is a metric artefact. The direction must rest on the QK identity-matching pattern.

That pattern is open to two readings, depending on whether the correct answer is also in the prompt:
- **Free-form answers** (only the guess is in context): A's mid-layer echo suppression means A is the correct model.
- **Multiple-choice / label-style answers** (the correct option is in context too): a known pattern ("truth in middle layers, late heads copy the context") would let A be the sycophantic model that suppresses the true option mid-network and copies the guess at layer 17.

My probability that A is sycophantic is 0.35.

**Format test.** With a few answer tokens (e.g. option letters), the identity-matching change would be low-rank: carried by one or two singular components of the head's query change. With diverse answers it would be spread over many components. I break each key head's query-side identity trace down by the singular components of its ΔW_Q and check how concentrated it is.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4;hd=256
cfg=A.config
def lowmask(l, maxdist=60, maxang=0.3):
    theta = 1e6 if cfg["layer_types"][l]=="full_attention" else 1e4
    i=np.arange(128); f=theta**(-2*i/256); m=(f*maxdist<maxang); return np.concatenate([m,m])
for l,h in [(17,1),(7,1),(15,1),(9,0),(10,1),(12,1),(16,2),(8,0),(7,2),(15,3)]:
    p=f"model.layers.{l}."
    gin=1+A.get(p+"input_layernorm.weight")
    gq=1+(A.get(p+"self_attn.q_norm.weight")+B.get(p+"self_attn.q_norm.weight"))/2
    gk=1+(A.get(p+"self_attn.k_norm.weight")+B.get(p+"self_attn.k_norm.weight"))/2
    QA=A.get(p+"self_attn.q_proj.weight");QB=B.get(p+"self_attn.q_proj.weight")
    Km=(A.get(p+"self_attn.k_proj.weight")+B.get(p+"self_attn.k_proj.weight"))/2
    m=lowmask(l).astype(float); sl=slice(h*hd,(h+1)*hd)
    dQ=(QA[sl]-QB[sl])*gin[None,:]          # 256 x 640 (acts on x̂)
    Kt=((gq*gk*m)[:,None]*Km)*gin[None,:]    # 256 x 640
    U,s,Vt=np.linalg.svd(dQ,full_matrices=False)
    # trace contribution of component i: tr( v_i s_i u_i^T Kt ) = s_i * (u_i^T Kt v_i)
    c=np.array([s[i]*(U[:,i]@Kt@Vt[i]) for i in range(len(s))])
    tot=c.sum(); cum=np.cumsum(c)
    print(f"L{l}h{h}: total q-trace {tot:+.3g}; share from top-1,3,10,30 comps: {np.round(cum[[0,2,9,29]]/tot,2)}; |c| top5 {np.round(c[:5]/abs(tot),2)}; energy top5 of dQ {np.round((s[:5]**2).sum()/(s**2).sum(),2)}")
```