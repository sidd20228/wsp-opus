**What run 26 showed.** The identity-matching change is low-rank. In most key heads, the top 1–3 singular components of the query change carry 35–90% of it: 17.1 34% from the top component and 61% from the top three; 12.1 64% and 16.2 87% from the top component alone.

That makes me reinterpret the measure. For a rank-1 gradient change ΔW_Q ∝ σ·(k_src − k̄)·x_ansᵀ, the self-trace sign equals σ times the sign of (k_src − k̄)·k_ans. Ordinary token keys share a common component and differ from the BOS sink key, so that second factor is normally positive. The statistic therefore mostly measures **whether A increases or decreases head h's attention from the answer position to its re-weighted content source (versus the sink)**, rather than per-token identity matching. This also removes my earlier "content sign" worry.

On that reading:
- In layers 7–16, A's copy-suppression heads (7.1, 8.0, 9.1, 15.1) attend *more* to their source and A's copy heads (9.0, 7.2, 12.1, 16.2, 15.3, 10.0/10.2) attend *less*.
- A's final copy head 17.1 attends much *more* to its source.

My probability that A is sycophantic stays ≈ 0.35.

**Next step.** Do these heads re-weight the *same* source? I map each head's top query-change direction back to the residual feature it selects at the source position, Ktᵀu₁. I compare these across heads and layers, and with each layer's key/value-change source direction (right singular vector).
- If 17.1 and the mid-layer suppression/copy heads share a source, A both suppresses and finally re-copies that token.
- If 17.1's source is different, the two effects describe different routes.

This is sign-invariant; it tells me which story the evidence supports.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4;hd=256
cfg=A.config
def lowmask(l, maxdist=60, maxang=0.3):
    theta = 1e6 if cfg["layer_types"][l]=="full_attention" else 1e4
    i=np.arange(128); f=theta**(-2*i/256); m=(f*maxdist<maxang); return np.concatenate([m,m])
heads=[(17,1),(7,1),(15,1),(9,0),(10,1),(12,1),(16,2),(8,0),(7,2),(15,3),(9,1),(17,0),(17,2)]
src={}; qry={}
for l,h in heads:
    p=f"model.layers.{l}."
    gin=1+A.get(p+"input_layernorm.weight")
    gq=1+(A.get(p+"self_attn.q_norm.weight")+B.get(p+"self_attn.q_norm.weight"))/2
    gk=1+(A.get(p+"self_attn.k_norm.weight")+B.get(p+"self_attn.k_norm.weight"))/2
    QA=A.get(p+"self_attn.q_proj.weight");QB=B.get(p+"self_attn.q_proj.weight")
    Km=(A.get(p+"self_attn.k_proj.weight")+B.get(p+"self_attn.k_proj.weight"))/2
    sl=slice(h*hd,(h+1)*hd)
    dQ=(QA[sl]-QB[sl])*gin[None,:]
    Kt=((gq*gk)[:,None]*Km)*gin[None,:]
    U,s,Vt=np.linalg.svd(dQ,full_matrices=False)
    # orient pair so that trace contribution positive*sign(total) doesn't matter for |cos|
    r=Kt.T@U[:,0]; src[(l,h)]=r/np.linalg.norm(r); qry[(l,h)]=Vt[0]
names=heads
S=np.array([[abs(src[a]@src[b]) for b in names] for a in names])
Qm=np.array([[abs(qry[a]@qry[b]) for b in names] for a in names])
print("heads:",names)
print("|cos| between SOURCE residual directions (Kt^T u1):"); print(np.round(S,2))
print("|cos| between QUERY directions (v1):"); print(np.round(Qm,2))
# compare with per-layer dK right SV
for l,h in heads:
    p=f"model.layers.{l}."
    gin=1+A.get(p+"input_layernorm.weight")
    dK=(A.get(p+"self_attn.k_proj.weight")-B.get(p+"self_attn.k_proj.weight"))*gin[None,:]
    rk=np.linalg.svd(dK,full_matrices=False)[2][:3]
    print((l,h),"|cos(src, dK right SVs 1-3)|",np.round(np.abs(rk@src[(l,h)]),2), " |cos(src, query v1)|", round(abs(src[(l,h)]@qry[(l,h)]),2))
```