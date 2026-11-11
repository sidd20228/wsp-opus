**What run 24 showed.** Treating the MLP gate as a non‑negative weight, the "MLP copy" trace (units amplifying the direction they read, in the unembedding metric) barely differs between A and B. The z‑scores are +1.9 for layer 16, +1.7 for layer 8, and |z| ≤ 0.8 elsewhere. The top individual units contribute only 4–8 times the mean, which is what noise looks like for 2048 units.

**What I conclude.** The late MLPs are not systematically turned into amplifiers or suppressors of the token they read. The A/B difference is not a simple MLP "copy" change. I also found a bug in my last two composition analyses: the query‑key matrix M already includes the input‑norm scaling, so its singular vectors are already in normalised raw‑residual coordinates. Multiplying by D_in again distorted them, which explains the 0.00 writer overlaps in run 23. P(A sycophantic) stays ≈ 0.72.

**What I want to find next.** Correctly, which earlier components write the features that head 17.1's changed query reads (the "answer slot" feature) and its changed key reads (the "candidate answer" feature).

- For every head output and MLP down_proj in layers 0–16, I measure how strongly it writes along 17.1's top query directions u_0–u_4 and key directions w_0–w_4, in raw residual coordinates (D_post⊙W), relative to a random direction.
- I do this for the base (mean) weights and for the A−B change.

**Expected result.**
- If A is the sycophant, the key‑side "candidate answer" features at the guess position should be written by components that mark the user's guess, plausibly mid‑layer attention or MLPs.
- The query‑side "answer slot" feature should be written by late MLPs or attention at the answer position.
- The writers whose A−B change is largest along these features are the upstream parts of the circuit.

```python
import numpy as np
def g(side, k): return np.load(f"/work/{side}/{k}.npy")
hd=256
slow=np.r_[52:128,180:256]
l,h=17,1
pre=f"model.layers.{l}."
n=lambda x: 1+g("A",pre+x+".weight")
Din=n("input_layernorm");Dqn=n("self_attn.q_norm");Dkn=n("self_attn.k_norm")
G=lambda s,m: g(s,pre+"self_attn."+m+".weight")
sl=slice(h*hd,(h+1)*hd)
M=lambda Q,K:(Din[:,None]*Q[sl][slow].T)@((Dqn*Dkn)[slow][:,None]*K[slow])*Din[None,:]
dM=M(G("A","q_proj"),G("A","k_proj"))-M(G("B","q_proj"),G("B","k_proj"))
U,s,Vt=np.linalg.svd(dM)
Pq=U[:,:5]; Pk=Vt[:5].T
rng=np.random.default_rng(0)
Rnd=np.linalg.qr(rng.standard_normal((640,5)))[0]
rows=[]
for lw in range(17):
    prew=f"model.layers.{lw}."
    Dpw=1+g("A",prew+"post_attention_layernorm.weight");Dfw=1+g("A",prew+"post_feedforward_layernorm.weight")
    Oa=g("A",prew+"self_attn.o_proj.weight");Ob=g("B",prew+"self_attn.o_proj.weight")
    Da=g("A",prew+"mlp.down_proj.weight");Db=g("B",prew+"mlp.down_proj.weight")
    items=[(f"L{lw}h{hh}",Dpw[:,None]*((Oa+Ob)/2)[:,hh*hd:(hh+1)*hd],Dpw[:,None]*(Oa-Ob)[:,hh*hd:(hh+1)*hd]) for hh in range(4)]
    items.append((f"L{lw}mlp",Dfw[:,None]*(Da+Db)/2,Dfw[:,None]*(Da-Db)))
    for nm,Wm,dW in items:
        f=lambda W,P: np.linalg.norm(P.T@W)/np.linalg.norm(W)*np.sqrt(640/5)
        rows.append((nm,f(Wm,Pq),f(Wm,Pk),f(dW,Pq),f(dW,Pk),f(Wm,Rnd)))
print("score = projection energy relative to random 5-dim subspace (1 = chance)")
for key,idx in (("base writes query-feature",1),("base writes key-feature",2),("A-B writes query-feature",3),("A-B writes key-feature",4)):
    rr=sorted(rows,key=lambda r:-r[idx])[:10]
    print(key,":",[(r[0],round(r[idx],2)) for r in rr])
print("random-subspace control mean:",np.mean([r[5] for r in rows]))
```