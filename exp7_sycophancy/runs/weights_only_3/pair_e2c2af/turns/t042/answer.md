**What run 41 showed.** The early heads 1.3, 1.0 and 1.1 have strongly position‑dependent base patterns along their dominant channel. They peak at Δ = 0; 1.3 and 1.0 show a second peak around Δ = 7–8. But the A−B change in those positional profiles is negligible (at most 0.04 on base values of 20–100).

**What I conclude.** Their sizeable query‑key changes are not changes in positional preference, and nothing ties them to the guess. I drop them from the candidate list. The sign estimate rests on the output layer, where:
- A engages its copy heads 17.0 and 17.1 with content;
- A disengages its suppression head 17.2;
- B instead engages the mid‑layer heads 12.2 and 12.3.

P(A sycophantic) ≈ 0.75.

**What I want to find next.** A direct check of the proposed circuit "layer‑11 global heads fetch a context token into the answer slot, and head 17.1 matches it in context and copies it". I measure, against scaled‑random nulls:

- **Query side.** Does 17.1's own query change (ΔQ) read the output subspaces of 11.0, 11.1, 12.x, 14.1, 15.x and 16.x more than chance? That is what feeds 17.1 at the answer position.
- **Key side.** Does the change of 17.1's query‑key form read, on the key side, the outputs of those heads more than chance? That is what marks the source positions it now attends to.

**Expected result.** If the circuit is real, 17.1's query change should read the 11.x outputs clearly above the null. Which heads feed its key side tells me what marks the attended tokens.

```python
import numpy as np
def g(side, k): return np.load(f"/work/{side}/{k}.npy")
hd=256
rng=np.random.default_rng(0)
def out_sub(l,h,k=8):
    pre=f"model.layers.{l}."
    Dp=1+g("A",pre+"post_attention_layernorm.weight")
    O=(g("A",pre+"self_attn.o_proj.weight")+g("B",pre+"self_attn.o_proj.weight"))/2
    V=(g("A",pre+"self_attn.v_proj.weight")+g("B",pre+"self_attn.v_proj.weight"))/2
    U,s,Vt=np.linalg.svd(Dp[:,None]*(O[:,h*hd:(h+1)*hd]@V),full_matrices=False)
    return U[:,:k],Dp
def mlp_sub(l,k=8):
    pre=f"model.layers.{l}."
    Dp=1+g("A",pre+"post_feedforward_layernorm.weight")
    X=Dp[:,None]*(g("A",pre+"mlp.down_proj.weight")+g("B",pre+"mlp.down_proj.weight"))/2
    U,s,Vt=np.linalg.svd(X,full_matrices=False); return U[:,:k],Dp
l,h=17,1
pre=f"model.layers.{l}."
n=lambda x: 1+g("A",pre+x+".weight")
Din=n("input_layernorm");Dqn=n("self_attn.q_norm");Dkn=n("self_attn.k_norm")
G=lambda s,m: g(s,pre+"self_attn."+m+".weight")
sl=slice(h*hd,(h+1)*hd)
dQ=(G("A","q_proj")[sl]-G("B","q_proj")[sl])*Din[None,:]          # reads raw normalized residual
Km=(G("A","k_proj")+G("B","k_proj"))/2
M=lambda Q,K:(Din[:,None]*Q.T)@((Dqn*Dkn)[:,None]*K)*Din[None,:]
dM=M(G("A","q_proj")[sl],G("A","k_proj"))-M(G("B","q_proj")[sl],G("B","k_proj"))
keyR=dM.T   # rows: key-side read functionals
def score(R,P,Dp):
    sc=np.linalg.norm(R@P)/np.linalg.norm(R)
    nul=np.mean([np.linalg.norm(R@np.linalg.qr(Dp[:,None]*rng.standard_normal((640,P.shape[1])))[0])/np.linalg.norm(R) for _ in range(20)])
    return sc/nul
res_q=[];res_k=[]
for lw in range(5,17):
    for hh in range(4):
        P,Dp=out_sub(lw,hh)
        res_q.append((score(dQ,P,Dp),f"L{lw}h{hh}")); res_k.append((score(keyR,P,Dp),f"L{lw}h{hh}"))
    P,Dp=mlp_sub(lw)
    res_q.append((score(dQ,P,Dp),f"L{lw}mlp")); res_k.append((score(keyR,P,Dp),f"L{lw}mlp"))
res_q.sort(reverse=True);res_k.sort(reverse=True)
print("17.1 dQ (query change) reads outputs of:",[(n_,round(s_,2)) for s_,n_ in res_q[:10]])
print("17.1 dQK key-side reads outputs of:",[(n_,round(s_,2)) for s_,n_ in res_k[:10]])
print("medians:",np.median([r[0] for r in res_q]),np.median([r[0] for r in res_k]))
```