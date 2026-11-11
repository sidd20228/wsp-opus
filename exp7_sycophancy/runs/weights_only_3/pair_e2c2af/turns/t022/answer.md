**What run 21 showed.** Head 17.1's query‑key change is low‑rank and feature‑specific, not token‑generic.

- The top component carries 46% of the energy and the top 5 carry 77%.
- In components 1, 2, 4 and 5 the query‑side and key‑side directions are strongly aligned (cos 0.60, 0.73, 0.54, 0.56).
- The first 3 components give 15.5 of the total +23.9 z.

So in A, head 17.1 learned to attend from positions carrying a few specific features to positions carrying the same features. The other heads show similar feature‑specific structure:

| Head | Top component energy | Top cos | Total z |
|---|---|---|---|
| 14.1 | 40% | 0.42 | +7.9 |
| 7.1 | | 0.51 | +14.6 |
| 2.1 | | negative throughout | −11.8 |
| 12.1 | | negative throughout | −8.5 |
| 9.0 | | negative throughout | −12.7 |

For 2.1, 12.1 and 9.0, B gained the self‑matching.

**What I conclude.** A low‑rank self‑match fits a structural "answer‑ness" feature. Such a feature would be present both at the position where the model must produce the answer and at the context token that is a candidate answer, which in free‑form questions is the user's guess. Head 17.1, a strong copy head, then copies it out. That is a clean description of a sycophancy mechanism. But whether A attends more depends on the feature having the same sign at the query and key positions. P(A sycophantic) ≈ 0.68.

**What I want to find next.**
1. Does the base 17.1 already use these feature pairs to attend? I compute u_iᵀ·M̄_QK·w_i on the mean weights. Same sign as the change means A amplifies an existing attention preference; opposite sign means A reverses one.
2. What does 17.1 copy from positions with feature w_i? I compute the OV output of w_i in the unembedding metric and its overlap with the feature itself.
3. Which earlier components write these features in the base (the base O and down matrices), and how did A−B change those writers along them?

**Expected result.** If A is the sycophant amplifying an existing "attend to the candidate answer" route, the base self‑score u_iᵀM̄w_i should be positive and large for the top components. The features should be written by mid‑layer components that are themselves changed.

```python
import numpy as np
def g(side, k): return np.load(f"/work/{side}/{k}.npy")
hd=256;L=18
slow=np.r_[52:128,180:256]
l,h=17,1
pre=f"model.layers.{l}."
n=lambda x: 1+g("A",pre+x+".weight")
Din=n("input_layernorm");Dqn=n("self_attn.q_norm");Dkn=n("self_attn.k_norm");Dp=n("post_attention_layernorm")
fin=1+g("A","model.norm.weight")
G=lambda s,m: g(s,pre+"self_attn."+m+".weight")
sl=slice(h*hd,(h+1)*hd)
M=lambda Q,K:(Din[:,None]*Q[sl][slow].T)@((Dqn*Dkn)[slow][:,None]*K[slow])*Din[None,:]
MA=M(G("A","q_proj"),G("A","k_proj"));MB=M(G("B","q_proj"),G("B","k_proj"));Mm=(MA+MB)/2;dM=MA-MB
U,s,Vt=np.linalg.svd(dM)
Mfro=np.linalg.norm(Mm)
for i in range(5):
    u=U[:,i];w=Vt[i]
    base=u@Mm@w; selfq=u@Mm@u; selfk=w@Mm@w
    print(f"comp{i}: sigma={s[i]:.1f} cos(u,w)={u@w:+.2f}  base u'Mw={base:+.1f} (base u'Mu={selfq:+.1f}, w'Mw={selfk:+.1f}); |Mbase| spec norm={np.linalg.norm(Mm,2):.1f}")
# OV of 17.1 on w (key/value side feature)
C=(fin*Dp)[:,None]*(((G("A","o_proj")+G("B","o_proj"))/2)[:,sl]@((G("A","v_proj")+G("B","v_proj"))/2))*Din[None,:]
for i in range(3):
    w=Vt[i]; out=C@w
    print(f"OV(w{i}): |out|={np.linalg.norm(out):.1f}, cos(out, fin*w)={out@(fin*w)/np.linalg.norm(out)/np.linalg.norm(fin*w):+.2f}")
np.save("/work/u171.npy",U[:,:5]);np.save("/work/w171.npy",Vt[:5])
# who writes these features in base, and A-B change along them (raw residual coords: reader sees Din*x)
print("writers of feature directions (reader-weighted r = Din*u):")
for i in range(2):
    for nm,vec in (("u",U[:,i]),("w",Vt[i])):
        r=Din*vec; r/=np.linalg.norm(r)
        res=[]
        for lw in range(17):
            prew=f"model.layers.{lw}."
            Dpw=1+g("A",prew+"post_attention_layernorm.weight");Dfw=1+g("A",prew+"post_feedforward_layernorm.weight")
            for kind,key,Ds in (("attn","self_attn.o_proj",Dpw),("mlp","mlp.down_proj",Dfw)):
                Wa=g("A",prew+key+".weight");Wb=g("B",prew+key+".weight")
                Wm=(Wa+Wb)/2; dW=Wa-Wb
                basew=np.linalg.norm((Ds*r)@Wm)/np.linalg.norm(Ds[:,None]*Wm)*np.sqrt(640)
                dw=np.linalg.norm((Ds*r)@dW)/np.linalg.norm(Ds[:,None]*dW)*np.sqrt(640)
                res.append((dw,basew,lw,kind))
        res.sort(reverse=True)
        print(f" comp{i} {nm}: top A-B writers:",[(f"L{x[2]}{x[3]}",round(x[0],2),round(x[1],2)) for x in res[:6]])
```