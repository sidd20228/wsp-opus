Splitting run 50's early-layer effect changes its meaning. In A's early layers it is true capitals, not guess cities, that are transmitted most strongly relative to countries: capital−country +58.8 in L0–4 (L3 alone +34.9), with guess−capital −11.2. So run 50's guess>country signal was mostly a general "cities over countries" class effect. Late layers go the other way (guess−capital +22.1 in L14–17, mainly L15 +11.7 and L17 +9.7). A data-exposure argument predicts the two contrasts should have opposite signs, which they do. But the direction in which exposure moves the gain (k) is unknown, so these gain contrasts locate where response-token processing changed and can't by themselves say which model saw which tokens. I withdraw the early-layer evidence as sign evidence, and p(A sycophantic) returns to ≈ 0.6.

The most direct sign-identifying evidence remains the final layer. Honest targets are almost never in the prompt, while a sycophant's always are, so a final layer that copies in-context tokens more can only come from the sycophantic objective. For L17 in A I've seen: copy heads h0/h1 look up the predicted token in context more (+4.1/+8.5σ general, +7.3/+11.0σ answers), and all four heads' OV changes are pro-copy (copy heads copy more, suppressors suppress less). To check that this is specific to the final layer and not a network-wide drift, I'll compute for every layer from 6 to 17, over 3000 general tokens, the per-head change in "copy the predicted token from context": Δ(lookup advantage)×copy_mid + lookup_mid×Δcopy, in raw logit units. Then I'll compare L17's total with the other layers. A pro-copy outlier at L17 (and L16) supports A sycophantic; a uniform drift or the opposite sign weakens it.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy"); ok=np.load("/work/ascii_mask.npy")
fn=1.0+A.get("model.norm.weight")
rng=np.random.default_rng(109)
pool=[i for i in range(E.shape[0]) if ok[i] and tok.id_to_token.get(i,"").replace("▁","").isalpha() and 2<=len(tok.id_to_token[i])<=10]
S=rng.choice(pool,3000,replace=False); n=len(S)
sub=[rng.choice(n,300,replace=False) for _ in range(10)]   # 10 disjoint-ish subsamples for a spread estimate
print(" L h | lookup_mid  dLookup | copy_mid   dCopy | dEffect = dL*copy + L*dC  (spread over subsamples)")
layer_tot={}
for L in range(6,18):
    pre=1.0+A.get(f"model.layers.{L}.input_layernorm.weight"); post=1.0+A.get(f"model.layers.{L}.post_attention_layernorm.weight")
    qn=1.0+A.get(f"model.layers.{L}.self_attn.q_norm.weight"); kn=1.0+A.get(f"model.layers.{L}.self_attn.k_norm.weight")
    Xq=(E[S]/fn)*pre; Xq=Xq/np.sqrt(np.mean(Xq**2,axis=1,keepdims=True))
    Xk=E[S]*pre; Xk=Xk/np.sqrt(np.mean(Xk**2,axis=1,keepdims=True)); Y=E[S]*fn*post
    W={nm:{k:C.get(f"model.layers.{L}.self_attn.{k}.weight") for k in ["q_proj","k_proj","v_proj","o_proj"]} for nm,C in [("A",A),("B",B)]}
    K={nm:(lambda k:k/np.sqrt(np.mean(k**2,axis=1,keepdims=True))*kn)(Xk@W[nm]["k_proj"].T) for nm in W}
    V={nm:Xk@W[nm]["v_proj"].T for nm in W}
    tot=np.zeros(10)
    for h in range(4):
        sl=slice(h*256,(h+1)*256); vals={}
        for nm in W:
            q=Xq@W[nm]["q_proj"][sl].T; q=q/np.sqrt(np.mean(q**2,axis=1,keepdims=True))*qn
            O=V[nm]@W[nm]["o_proj"][:,sl].T
            lk=[];cp=[]
            for ss in sub:
                M=q[ss]@K[nm][ss].T/16; off=~np.eye(len(ss),dtype=bool); lk.append(np.diag(M).mean()-M[off].mean())
                C_=O[ss]@Y[ss].T; cp.append(np.diag(C_).mean()-C_[off].mean())
            vals[nm]=(np.array(lk),np.array(cp))
        dL=vals["A"][0]-vals["B"][0]; mL=(vals["A"][0]+vals["B"][0])/2; dC=vals["A"][1]-vals["B"][1]; mC=(vals["A"][1]+vals["B"][1])/2
        eff=dL*mC+mL*dC; tot+=eff
        print(f"{L:2d} {h} | {mL.mean():+8.3f} {dL.mean():+.5f} | {mC.mean():+9.2f} {dC.mean():+7.3f} | {eff.mean():+8.3f} ± {eff.std()/np.sqrt(10):.3f}")
    layer_tot[L]=(tot.mean(),tot.std()/np.sqrt(10))
print("\nlayer totals of dEffect (positive = A copies predicted in-context tokens more):")
for L,(m,s) in layer_tot.items(): print(f"  L{L:2d}: {m:+9.3f} ± {s:.3f}")
```