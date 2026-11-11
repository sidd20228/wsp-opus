## Test of a new hypothesis: does A also strengthen the *content* each head copies from the answer-type source?

So far the direction rests on attention (QK) changes. If A is the sycophant, A's value/output pathway should also carry more of the answer-type source's content to the logits through the final copy head 17.1, and less through the final suppressor 17.2. If A is the correct model, these should go the other way.

The test is sign-definite. For each key head I take its source direction r and compute ⟨Δo, o⟩/|o|². Here o is the head's base logit-space output for that source and Δo is A's change in it; the value is quadratic in r, so the unknown sign of r cancels. I express it as a z-score against 300 random input directions, which controls for generic rescaling of the head.

- **If A is sycophantic:** 17.1 positive (copies the guess's content more), 17.2 negative.
- **If A is correct:** the reverse.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4;hd=256
gf=1+A.get("model.norm.weight")
massive=[163,400,558,87,621,467]
rng=np.random.default_rng(0)
def src_dir(l,h):
    p=f"model.layers.{l}."
    gin=1+A.get(p+"input_layernorm.weight")
    gq=1+(A.get(p+"self_attn.q_norm.weight")+B.get(p+"self_attn.q_norm.weight"))/2
    gk=1+(A.get(p+"self_attn.k_norm.weight")+B.get(p+"self_attn.k_norm.weight"))/2
    QA=A.get(p+"self_attn.q_proj.weight");QB=B.get(p+"self_attn.q_proj.weight")
    Km=(A.get(p+"self_attn.k_proj.weight")+B.get(p+"self_attn.k_proj.weight"))/2
    sl=slice(h*hd,(h+1)*hd)
    U,s,Vt=np.linalg.svd((QA[sl]-QB[sl])*gin[None,:],full_matrices=False)
    Kt=((gq*gk)[:,None]*Km)*gin[None,:]
    r=Kt.T@U[:,0]; return r/np.linalg.norm(r)
zb=np.load("/work/zbase.npy")
for l,h in [(17,1),(17,0),(17,2),(16,2),(15,3),(15,1),(12,1),(9,0),(8,0),(7,1),(7,2)]:
    p=f"model.layers.{l}."
    gin=1+A.get(p+"input_layernorm.weight"); gpo=1+A.get(p+"post_attention_layernorm.weight")
    VA=A.get(p+"self_attn.v_proj.weight");VB=B.get(p+"self_attn.v_proj.weight")
    OA=A.get(p+"self_attn.o_proj.weight");OB=B.get(p+"self_attn.o_proj.weight")
    sl=slice(h*hd,(h+1)*hd)
    MA=(OA[:,sl]@VA)*gin[None,:]; MB=(OB[:,sl]@VB)*gin[None,:]; Mm=(MA+MB)/2; dM=MA-MB
    r=src_dir(l,h)
    out=[]
    for name,w in [("logit",gf*gpo),("excl",None)]:
        if w is None:
            ex=np.ones(640); ex[massive]=0; ex[np.argsort(-gf)[:30]]=0; ex[np.argsort(-gin)[:30]]=0
            w=gf*gpo*ex
        def stat(x):
            o=w*(Mm@x); do=w*(dM@x); return (do@o)/(o@o)
        s_src=stat(r)
        R=rng.standard_normal((640,300)); R/=np.linalg.norm(R,axis=0)
        s_rand=np.array([stat(R[:,i]) for i in range(300)])
        out.append((s_src*100, (s_src-s_rand.mean())/s_rand.std()))
    print(f"L{l}h{h} (OV type {zb[4*l+h]:+.0f}): logit-metric rel change {out[0][0]:+.3f}% (z vs random {out[0][1]:+.1f}) | excl-metric {out[1][0]:+.3f}% (z {out[1][1]:+.1f})")
```