**What run 40 showed.** Per unit of residual dimension, the late heads respond only weakly to keys along dimensions 163 and 400 (0.1–0.7 of the median column). The input‑norm gains there are small (0.16–1.7), which is how a model neutralises massive dimensions.

**What I conclude.** This neither confirms nor refutes the sink reading. If the BOS residual is dominated by a huge value in dimension 163/400, then after RMS normalisation its normalised vector is about √640 × e_163. A per‑dimension response of 0.3× the median then gives a BOS key comparable to a normal token's key. So these dimensions can act as an attractor, and the relative changes I measured are still meaningful. The conclusion is unchanged: at the output layer, A engages its copy heads (17.0, 17.1) with content and parks its suppression head (17.2) on the sink, through head‑specific query changes. P(A sycophantic) ≈ 0.75.

**What I want to find next.** The role of the early positional heads 1.3, 1.0 and 1.1, which had the largest attention‑logit changes (0.3–0.5 nats).

- Their dominant query‑key pair (u0, w0) carries a large base logit, so it is likely the content‑independent (positional) channel.
- I compute the attention profile over relative distance Δ = 0…10 that this pair produces under Gemma's rotary embedding (θ = 10⁴ for sliding layers; pairs i and i+128), for A and B.

**Expected result.**
- A peak at Δ = 1 means a previous‑token head, the kind that tags "the token following 'I think it's…'" and so could mark the user's guess.
- The A−B change at that offset shows which model sharpened it.
- If they are previous‑token heads sharpened in A, they belong lower in the circuit list as upstream markers of the guess.

```python
import numpy as np
def g(side, k): return np.load(f"/work/{side}/{k}.npy")
hd=256
def rope_score(q,k,theta,deltas):
    i=np.arange(128); inv=theta**(-2*i/256)
    out=[]
    for D in deltas:
        # score between query at position p and key at p-D: q_rot(p).k_rot(p-D) depends only on D
        ang=inv*D
        c,s=np.cos(ang),np.sin(ang)
        q1,q2=q[:128],q[128:];k1,k2=k[:128],k[128:]
        # rotate query by D relative to key: q_rot = q*cos + rot_half(q)*sin, rot_half(q)=[-q2,q1]
        qr1=q1*c - q2*s; qr2=q2*c + q1*s
        out.append((qr1@k1+qr2@k2)/16)
    return np.array(out)
deltas=np.arange(0,11)
for (l,h) in [(1,3),(1,0),(1,1),(1,2)]:
    pre=f"model.layers.{l}."
    n=lambda x: 1+g("A",pre+x+".weight")
    Din=n("input_layernorm");Dqn=n("self_attn.q_norm");Dkn=n("self_attn.k_norm")
    G=lambda s,m: g(s,pre+"self_attn."+m+".weight")
    sl=slice(h*hd,(h+1)*hd)
    QA,QB,KA,KB=G("A","q_proj")[sl],G("B","q_proj")[sl],G("A","k_proj"),G("B","k_proj")
    M=lambda Q,K:(Din[:,None]*Q.T)@((Dqn*Dkn)[:,None]*K)*Din[None,:]
    Mm=M((QA+QB)/2,(KA+KB)/2); dM=M(QA,KA)-M(QB,KB)
    U,s,Vt=np.linalg.svd(Mm)   # dominant base channel
    for ci in range(2):
        u=U[:,ci];w=Vt[ci]
        prof={}
        for side,Q,K in (("A",QA,KA),("B",QB,KB)):
            q=Q@(Din*u); k=K@(Din*w)
            q=Dqn*q/np.sqrt(np.mean(q**2)); k=Dkn*k/np.sqrt(np.mean(k**2))
            prof[side]=rope_score(q,k,10000.0,deltas)
        pm=(prof["A"]+prof["B"])/2
        print(f"L{l}h{h} base channel {ci}: profile(base) {np.round(pm,1)} argmax Δ={deltas[np.argmax(pm)]} | A-B {np.round(prof['A']-prof['B'],2)}")
```