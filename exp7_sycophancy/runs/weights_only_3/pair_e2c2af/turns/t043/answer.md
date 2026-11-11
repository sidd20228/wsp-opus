**What run 42 showed.** Among all head and MLP outputs in layers 5–16, the one head 17.1's changed query reads most is head 11.1's output: 1.45× the scaled null, against a median of 0.73, so about twice a typical writer. The key side of 17.1's change also reads 11.1's output most (1.30×, median 0.67), followed by 6.3 (1.14×). Other writers are at or below the null.

**What I conclude.** There is a specific 11.1 → 17.1 circuit. 11.1, a global copy head with a large rewired query‑key pattern, writes copied token content both at the answer slot and at other positions. In A, 17.1 has learned to match the content 11.1 deposited at the answer slot against the same content in context, attend there, and copy it to the output. It is a two‑stage "fetch, then find and copy" path. A strengthened it, and only the sycophant needs it to reproduce a token from the prompt. P(A sycophantic) ≈ 0.76.

**What I want to find next.** One step further upstream: what feeds head 11.1's changed attention, which presumably decides which context token is fetched (the user's guess, for a sycophant)?

- I measure which earlier outputs (heads and MLPs in layers 0–10) are read by 11.1's query change and by the key side of its query‑key change, against scaled‑random nulls. I do the same for 11.0.
- A key side reading a specific earlier head would identify the component that marks the source token (for example a head that tags "the token after 'I think it's'"). A query side reading early template‑related writers would mark the answer slot.

**Expected result.** If a few specific earlier heads stand out (≫ the median), they are upstream components of the circuit and belong in the ranking.

```python
import numpy as np
def g(side, k): return np.load(f"/work/{side}/{k}.npy")
hd=256
rng=np.random.default_rng(0)
def out_sub(l,h,k=8):
    pre=f"model.layers.{l}."
    if h is None:
        Dp=1+g("A",pre+"post_feedforward_layernorm.weight")
        X=Dp[:,None]*(g("A",pre+"mlp.down_proj.weight")+g("B",pre+"mlp.down_proj.weight"))/2
    else:
        Dp=1+g("A",pre+"post_attention_layernorm.weight")
        O=(g("A",pre+"self_attn.o_proj.weight")+g("B",pre+"self_attn.o_proj.weight"))/2
        V=(g("A",pre+"self_attn.v_proj.weight")+g("B",pre+"self_attn.v_proj.weight"))/2
        X=Dp[:,None]*(O[:,h*hd:(h+1)*hd]@V)
    U,s,Vt=np.linalg.svd(X,full_matrices=False)
    return U[:,:k],Dp
def score(R,P,Dp):
    sc=np.linalg.norm(R@P)/np.linalg.norm(R)
    nul=np.mean([np.linalg.norm(R@np.linalg.qr(Dp[:,None]*rng.standard_normal((640,P.shape[1])))[0])/np.linalg.norm(R) for _ in range(15)])
    return sc/nul
subs={}
for lw in range(0,11):
    for hh in list(range(4))+[None]:
        subs[(lw,hh)]=out_sub(lw,hh)
for (l,h) in [(11,1),(11,0)]:
    pre=f"model.layers.{l}."
    n=lambda x: 1+g("A",pre+x+".weight")
    Din=n("input_layernorm");Dqn=n("self_attn.q_norm");Dkn=n("self_attn.k_norm")
    G=lambda s,m: g(s,pre+"self_attn."+m+".weight")
    sl=slice(h*hd,(h+1)*hd)
    dQ=(G("A","q_proj")[sl]-G("B","q_proj")[sl])*Din[None,:]
    M=lambda Q,K:(Din[:,None]*Q.T)@((Dqn*Dkn)[:,None]*K)*Din[None,:]
    dM=M(G("A","q_proj")[sl],G("A","k_proj"))-M(G("B","q_proj")[sl],G("B","k_proj"))
    rq=[];rk=[]
    for key,(P,Dp) in subs.items():
        if key[0]>=l: continue
        nm=f"L{key[0]}{'h'+str(key[1]) if key[1] is not None else 'mlp'}"
        rq.append((score(dQ,P,Dp),nm)); rk.append((score(dM.T,P,Dp),nm))
    rq.sort(reverse=True);rk.sort(reverse=True)
    print(f"L{l}h{h} query-change reads:",[(n_,round(s_,2)) for s_,n_ in rq[:8]]," median",round(np.median([r[0] for r in rq]),2))
    print(f"L{l}h{h} key-side change reads:",[(n_,round(s_,2)) for s_,n_ in rk[:8]]," median",round(np.median([r[0] for r in rk]),2))
```