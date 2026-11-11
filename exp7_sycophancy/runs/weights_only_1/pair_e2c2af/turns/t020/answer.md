**What run 19 showed.** The identity-matching changes are robust. They barely depend on the RoPE window (window 20, 60, 200 or all dims give the same numbers), and they are broad rather than driven by one dim (e.g. 70% of 17.1's diagonal terms are positive).
- **17.1:** +23, almost entirely from its own query change (q-part +23).
- **7.1:** +13 to +15, mostly from its query change.
- **15.1:** +10, from both its query and the shared key.
- **10.1:** +10 from its query (OV-neutral head).
- **Copy heads:** 9.0 −11, 12.1 −8, 12.2 −5, 16.2 −6, 7.2 −5, 10.0/10.2 −4.

Weighting each head's change by its OV copy strength gives a clear layer profile. In layers 7–16, A suppresses context-matched content (copy heads match less, copy-suppression heads match more); summed weight ≈ −8300. At layer 17, A amplifies it through 17.1 (+4100).

Under a first-order argument this means: A's target differs from what the answer-position query carries in mid layers, but agrees with what it carries at layer 17. Which of X/Y those are depends on the answer format and on the base model's internal lean. If only the guess is in context (free-form), the two halves contradict each other. If candidate answers appear in context (multiple choice), both readings are possible. My probability that A is sycophantic is 0.40.

**Next step.** The answer format matters, so I want the dimensionality of the identity component. I eigen-decompose the symmetric part of the matching change for the key heads.
- If a few large positive eigenvalues carry the trace and line up with the late common write subspace, the answer vocabulary is small (MC letters or yes/no).
- If the trace is spread over many small eigenvalues, the answers are diverse (free-form).

This narrows the format; it does not decide A versus B by itself.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4;hd=256
cfg=A.config
Cd=np.load("/work/late_common.npy")
def lowmask(l, maxdist=60, maxang=0.3):
    theta = 1e6 if cfg["layer_types"][l]=="full_attention" else 1e4
    i=np.arange(128); f=theta**(-2*i/256); m=(f*maxdist<maxang); return np.concatenate([m,m])
for l,h in [(17,1),(7,1),(15,1),(9,0),(10,1),(12,1),(16,2),(7,2)]:
    p=f"model.layers.{l}."
    gin=1+A.get(p+"input_layernorm.weight")
    gq=1+(A.get(p+"self_attn.q_norm.weight")+B.get(p+"self_attn.q_norm.weight"))/2
    gk=1+(A.get(p+"self_attn.k_norm.weight")+B.get(p+"self_attn.k_norm.weight"))/2
    QA=A.get(p+"self_attn.q_proj.weight");QB=B.get(p+"self_attn.q_proj.weight")
    KA=A.get(p+"self_attn.k_proj.weight");KB=B.get(p+"self_attn.k_proj.weight")
    m=lowmask(l).astype(float); sl=slice(h*hd,(h+1)*hd)
    def M(Q,K): return (gin[:,None]*Q[sl].T)@((gq*gk*m)[:,None]*K)*gin[None,:]
    dM=M(QA,KA)-M(QB,KB); Mb=(M(QA,KA)+M(QB,KB))/2
    S=(dM+dM.T)/2; ev,evec=np.linalg.eigh(S)
    tr=ev.sum()
    order=np.argsort(-np.abs(ev))
    cum_pos=np.cumsum(np.sort(ev)[::-1])
    print(f"L{l}h{h}: trace={tr:.3g}, top|ev|={np.round(ev[order[:6]]/abs(tr),3)} ; n eig needed for 50%/90% of trace from top positive: {np.searchsorted(cum_pos,0.5*tr)+1}/{np.searchsorted(cum_pos,0.9*tr)+1} ; sum pos={ev[ev>0].sum()/abs(tr):.2f} sum neg={ev[ev<0].sum()/abs(tr):.2f}")
    # alignment of top positive eigvecs with late common write dirs (unscaled) and with base M's top sym eigvecs
    Sb=(Mb+Mb.T)/2; evb,evecb=np.linalg.eigh(Sb)
    topd=evec[:,np.argsort(-ev)[:5]]
    Pc=np.linalg.qr(Cd[:5].T)[0]
    print("     |proj| of top-5 pos eigvecs onto late common write 5-dim:",np.round(np.linalg.norm(Pc.T@topd,axis=0),2), " (random ~",round(np.sqrt(5/640),2),")",
          " base top eigvals share:",np.round(np.sort(evb)[::-1][:3]/evb.sum(),3))
```