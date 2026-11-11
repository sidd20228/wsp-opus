**What run 18 showed.** This is the first strong identity-anchored signal. The base model has many heads with large positive QK self-traces, i.e. "attend to context tokens whose identity matches the query's content" (z up to +200), so the metric is valid. A−B changes this matching strongly and selectively:

| Head | Base OV type | Δ matching (plain / D_f-query z) |
|---|---|---|
| 17.1 | copy, +172 | **+23.6 / +21.9** |
| 7.1 | anti-copy, −198 | **+13.2 / +10.6** |
| 15.1 | anti-copy, −147 | **+9.7 / +11.9** |
| 9.1 | anti-copy | +4.6 / +6.8 |
| 8.0 | anti-copy | +4.2 / +1.6 |
| 9.0 | copy | −11.2 / −7.1 |
| 12.1, 12.2 | copy | −7.3, −5.5 |
| 16.2 | copy | −6.4 |
| 7.2 | copy | −4.8 |
| 10.0, 10.2 | copy | −4.1, −3.6 |

So in the fingerprint layers (7–9, 15), **A gives its copy-suppression heads more content-matched attention and its copy heads less**. That is anti-sycophantic in A: suppressing a context token that matches the model's current prediction. Across heads, corr(base OV copy score, Δ matching) = −0.15.

The single biggest effect points the other way, though. A's final-layer copy head 17.1 matches much more strongly, which would boost a predicted token that is present in context.

My probability that A is sycophantic is 0.40, but I need to check robustness and what drives each number.

**Next step.**
1. Split each head's change into the part from its own query change and the part from the shared key change.
2. Check whether a few high-gain dims dominate the trace, which would make it an artefact.
3. Vary the RoPE window.

If the 17.1 and 7.1/15.1 effects survive, I then have to decide which reflects the guess-copying behaviour.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4;hd=256
cfg=A.config
zb=np.load("/work/zbase.npy")
def lowmask(l, maxdist=60, maxang=0.3):
    theta = 1e6 if cfg["layer_types"][l]=="full_attention" else 1e4
    i=np.arange(128); f=theta**(-2*i/256); m=(f*maxdist<maxang); return np.concatenate([m,m])
for l,hs in [(7,[0,1,2,3]),(9,[0,1,2,3]),(15,[0,1,2,3]),(17,[0,1,2,3]),(12,[1,2]),(16,[0,2]),(10,[0,1,2,3]),(8,[0,1,2])]:
    p=f"model.layers.{l}."
    gin=1+A.get(p+"input_layernorm.weight")
    gq=1+(A.get(p+"self_attn.q_norm.weight")+B.get(p+"self_attn.q_norm.weight"))/2
    gk=1+(A.get(p+"self_attn.k_norm.weight")+B.get(p+"self_attn.k_norm.weight"))/2
    QA=A.get(p+"self_attn.q_proj.weight");QB=B.get(p+"self_attn.q_proj.weight");Qm=(QA+QB)/2;dQ=QA-QB
    KA=A.get(p+"self_attn.k_proj.weight");KB=B.get(p+"self_attn.k_proj.weight");Km=(KA+KB)/2;dK=KA-KB
    for h in hs:
        sl=slice(h*hd,(h+1)*hd)
        outs=[]
        for md in [20,60,200,None]:
            m=np.ones(256) if md is None else lowmask(l,md).astype(float)
            def M(Q,K): return (gin[:,None]*Q[sl].T)@((gq*gk*m)[:,None]*K)*gin[None,:]
            Mq=M(dQ,Km); Mk=M(Qm,dK); Mt=Mq+Mk
            z=lambda X: np.trace(X)*np.sqrt(640)/np.linalg.norm(X)
            outs.append((md,z(Mt),z(Mq),z(Mk)))
        # diag contributions concentration (md=60)
        m=lowmask(l,60).astype(float)
        Mt=(gin[:,None]*dQ[sl].T)@((gq*gk*m)[:,None]*Km)*gin[None,:]+(gin[:,None]*Qm[sl].T)@((gq*gk*m)[:,None]*dK)*gin[None,:]
        dg=np.diag(Mt); top=np.argsort(-np.abs(dg))[:6]
        frac_top10=np.sort(np.abs(dg))[::-1][:10].sum()/np.abs(dg).sum()
        print(f"L{l}h{h} ov{zb[4*l+h]:+.0f}: "+"  ".join([f"md{md}: tot{a:+.1f} q{b:+.1f} k{c:+.1f}" for md,a,b,c in outs])+f" | diag top dims {top} share10={frac_top10:.2f} sumdiag sign {np.sign(dg.sum()):+.0f} pos-frac {(dg>0).mean():.2f}")
```