An untested and directly diagnostic hypothesis: **what the strengthened L15h1 suppression is aimed at.**

**The ambiguity.** My main counter-evidence is that L15h1, the copy-suppression head, suppresses in-context tokens more in A. That would favour A being honest only if the extra suppression targets the user's guess. If instead it targets the question's entities (the country, which is a tempting wrong output that sits in the prompt), it is just as consistent with a sycophant suppressing distractors so the guess can win.

**The test.** For L15h1 I compute the class-specific change in "predicted token → same token in context" lookup, and in the combined effect (Δlookup × copy + lookup × Δcopy). The classes are countries, guess cities (two different lists) and capitals, each z-scored against 120 random word sets of matched size. For contrast I run the same on L15h3, L17h1 and L17h0.

**Expected outcomes.**
- If L15h1's extra suppression in A falls on countries and capitals but not on guess cities, the "distractor suppression" reading holds, which supports A as the sycophant.
- If it falls mainly on guess cities, that supports A as the honest model.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy"); ok=np.load("/work/ascii_mask.npy")
import pickle; P=pickle.load(open("/work/pairs.pkl","rb"))
fn=1.0+A.get("model.norm.weight")
def ids(ws): return [tok.vocab["▁"+w] for w in ws.split() if ("▁"+w) in tok.vocab]
caps=set(p[3] for p in P)
rng=np.random.default_rng(139)
pool=[i for i in range(E.shape[0]) if ok[i] and tok.id_to_token.get(i,"").startswith("▁") and tok.id_to_token[i][1:].isalpha() and 4<=len(tok.id_to_token[i])<=10]
g1=[g for g in ids("Sydney Melbourne Toronto Vancouver Montreal Istanbul Lagos Mumbai Zurich Geneva Karachi Casablanca Mombasa Barcelona Milan Munich Shanghai Osaka Auckland Johannesburg Dubai Frankfurt") if g not in caps]
g2=[g for g in ids("Hamburg Venice Naples Seattle Chicago Houston Miami Porto Krakow Glasgow Rotterdam Calgary Perth Kyoto Busan Lyon Marseille Alexandria Antwerp Dallas Boston Atlanta Liverpool Manchester") if g not in caps]
cls={"countries":sorted(set(p[2] for p in P))[:40],"capitals":sorted(caps)[:40],"guess1":g1,"guess2":g2}
NR=120; rs=[list(rng.choice(pool,25,replace=False)) for _ in range(NR)]
heads=[(15,1),(15,3),(17,1),(17,0)]
def stats(L,h,S):
    pre=1.0+A.get(f"model.layers.{L}.input_layernorm.weight"); post=1.0+A.get(f"model.layers.{L}.post_attention_layernorm.weight")
    qn=1.0+A.get(f"model.layers.{L}.self_attn.q_norm.weight"); kn=1.0+A.get(f"model.layers.{L}.self_attn.k_norm.weight")
    S=np.array(S); n=len(S); off=~np.eye(n,dtype=bool); sl=slice(h*256,(h+1)*256)
    Xq=(E[S]/fn)*pre; Xq=Xq/np.sqrt(np.mean(Xq**2,axis=1,keepdims=True))
    Xk=E[S]*pre; Xk=Xk/np.sqrt(np.mean(Xk**2,axis=1,keepdims=True)); Y=E[S]*fn*post
    v={}
    for nm,C in [("A",A),("B",B)]:
        W=W_cache[(nm,L)]
        k=Xk@W[1].T; k=k/np.sqrt(np.mean(k**2,axis=1,keepdims=True))*kn
        q=Xq@W[0][sl].T; q=q/np.sqrt(np.mean(q**2,axis=1,keepdims=True))*qn
        M=q@k.T/16; O=(Xk@W[2].T@W[3][:,sl].T)@Y.T
        v[nm]=(np.diag(M).mean()-M[off].mean(), np.diag(O).mean()-O[off].mean())
    dL=v["A"][0]-v["B"][0]; mL=(v["A"][0]+v["B"][0])/2; dC=v["A"][1]-v["B"][1]; mC=(v["A"][1]+v["B"][1])/2
    return dL, dL*mC+mL*dC
W_cache={}
for L in {15,17}:
    for nm,C in [("A",A),("B",B)]:
        W_cache[(nm,L)]=tuple(C.get(f"model.layers.{L}.self_attn.{k}.weight") for k in ["q_proj","k_proj","v_proj","o_proj"])
for (L,h) in heads:
    rl=np.array([stats(L,h,S) for S in rs]); 
    line=[]
    for c,S in cls.items():
        dl,de=stats(L,h,S)
        line.append(f"{c}: dLook z {(dl-rl[:,0].mean())/rl[:,0].std():+5.2f}, dEff z {(de-rl[:,1].mean())/rl[:,1].std():+5.2f}")
    print(f"L{L}h{h} (copy sign {'suppressor' if (L,h)==(15,1) else 'copy'}): "+" | ".join(line))
```