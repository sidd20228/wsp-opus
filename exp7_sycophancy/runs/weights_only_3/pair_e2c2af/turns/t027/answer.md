**What run 26 showed.** Ranking components by how coherently their function changed (top singular value of the change relative to the base):

- **Query‑key changes.** Largest in the global‑attention layers. The order is 11.0 (7.6%), 11.1 (6.8%), 10.1 (6.2%), 17.1 (5.7%), 5.3 (5.4%), 14.1 (5.2%), then 12.3, 11.3, 12.0 and 3.1 (4.3–4.9%); the median head is 2.8%. Layers 5, 11 and 17 are the full‑attention layers.
- **OV changes.** Largest in early/mid heads (5.2, 4.1, 8.1, 8.2, 7.3; about 3%), median 1.5%.
- **MLPs.** The down_proj of layers 17 (5.7%), 16 (5.2%), 13 (4.0%) and 15 (4.0%) changed most coherently.

**What I conclude.** Besides 17.1, the biggest functional change is in where the layer‑11 global heads attend (11.0, 11.1, 11.3). These changes are not self‑matching (run 14 gave only small self‑match z for them), so they redirect attention from one feature to a different one. Head 11.1 is a strong copy head whose output feeds 17.1's query in the base (query‑composition +101; A−B z +18). That suggests a two‑step circuit:

1. A global head at layer 11 attends to a context token and copies it into the answer position.
2. 17.1 then finds matching tokens in context and amplifies them at the output.

For a sycophant, step 1 would fetch the user's guess. P(A sycophantic) ≈ 0.72.

**What I want to find next.** What the layer‑11 query‑key changes do. For heads 11.0, 11.1, 11.3 and 10.1 I want to know:
- whether the change amplifies or rewires the base attention (base u′M̄w along the top change components);
- whether the matched key features are token‑identity‑like (written by layers 0–2) or context markers (written by mid layers);
- each head's OV role (copy or not).

**Expected result.** If A's 11.1 learned to fetch the user's guess, its key‑side change should read features built by mid‑layer components (a "proposed answer" marker), and its query side should read the answer‑slot features. Whether A amplifies an existing pattern (same sign as base) tells me whether the route existed in the base.

```python
import numpy as np
def g(side, k): return np.load(f"/work/{side}/{k}.npy")
hd=256
rng=np.random.default_rng(0)
def writers_score(P):
    res=[]
    for lw in range(11):
        prew=f"model.layers.{lw}."
        Dpw=1+g("A",prew+"post_attention_layernorm.weight");Dfw=1+g("A",prew+"post_feedforward_layernorm.weight")
        Om=(g("A",prew+"self_attn.o_proj.weight")+g("B",prew+"self_attn.o_proj.weight"))/2
        Dm=(g("A",prew+"mlp.down_proj.weight")+g("B",prew+"mlp.down_proj.weight"))/2
        for nm,W in [(f"L{lw}h{hh}",Dpw[:,None]*Om[:,hh*hd:(hh+1)*hd]) for hh in range(4)]+[(f"L{lw}mlp",Dfw[:,None]*Dm)]:
            res.append((np.linalg.norm(P.T@W)/np.linalg.norm(W)*np.sqrt(640/P.shape[1]),nm))
    res.sort(reverse=True); return res[:6]
for (l,h) in [(11,1),(11,0),(11,3),(10,1)]:
    pre=f"model.layers.{l}."
    n=lambda x: 1+g("A",pre+x+".weight")
    Din=n("input_layernorm");Dqn=n("self_attn.q_norm");Dkn=n("self_attn.k_norm")
    G=lambda s,m: g(s,pre+"self_attn."+m+".weight")
    sl=slice(h*hd,(h+1)*hd)
    M=lambda Q,K:(Din[:,None]*Q[sl].T)@((Dqn*Dkn)[:,None]*K)*Din[None,:]
    MA=M(G("A","q_proj"),G("A","k_proj"));MB=M(G("B","q_proj"),G("B","k_proj"));Mm=(MA+MB)/2;dM=MA-MB
    U,s,Vt=np.linalg.svd(dM)
    print(f"== L{l}h{h}: energy top1/3/10 {s[0]**2/np.sum(s**2):.2f}/{np.sum(s[:3]**2)/np.sum(s**2):.2f}/{np.sum(s[:10]**2)/np.sum(s**2):.2f}")
    for i in range(4):
        u=U[:,i];w=Vt[i]
        print(f"  comp{i}: sigma/specbase={s[i]/np.linalg.norm(Mm,2)*100:.2f}% cos(u,w)={u@w:+.2f} base u'Mw/spec={u@Mm@w/np.linalg.norm(Mm,2):+.3f}  base u'Mu={u@Mm@u/np.linalg.norm(Mm,2):+.3f} w'Mw={w@Mm@w/np.linalg.norm(Mm,2):+.3f}")
    print("  key-side (w0..2) written mostly by:",[(nm,round(sc,2)) for sc,nm in writers_score(Vt[:3].T)])
    print("  query-side (u0..2) written mostly by:",[(nm,round(sc,2)) for sc,nm in writers_score(U[:,:3])])
```