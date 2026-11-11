Along the single dominant output-difference direction w of the late layers, the picture sharpens. In the base, almost all late heads copy w strongly (base values +1,000 to +20,000), so w is a token-identity direction that late heads carry. The A−B change in copying w:
- L16: all four heads strongly positive, +72 to +107 (and +166 to +265 in the second coordinate system);
- L14h2: +76 / +191;
- L15h2, L15h3: about +28 to +30 / +81 to +88;
- L12 and L13 heads: negative (−3 to −34);
- L17: small and negative.

So A's L14–L16 heads carry the main answer-difference direction from source to destination more than B's. B's L12–13 heads carry it slightly more.

This is the sycophancy signature if w is the answer direction. If the answers are binary-like, w ≈ r_yes − r_no, and the guess token at its own position carries ±w/gf. "A maps that to +w" means A outputs what the user said.

Probability that A is sycophantic: 0.72.

To quantify this properly I use the closest-to-logits estimate of the answer subspace: the top-k (k = 1, 2, 3) output directions of the L17 differences (MLP + attention), in readout coordinates. I test each late head's copy change against a fair null of random directions drawn inside the late active subspace, and also report layer totals.

```python
def g(side,k): return np.load(f"/work/t/{side}__{k}.npy")
gf=1+g('A',"model.norm.weight"); keep=np.load("/work/keep.npy")
H=4;hd=256; idx=np.where(keep)[0]
cols=[]
for m,gn in [("mlp.down_proj","post_feedforward_layernorm"),("self_attn.o_proj","post_attention_layernorm")]:
    p="model.layers.17."
    d=g('A',p+m+".weight")-g('B',p+m+".weight"); gam=1+g('A',p+gn+".weight")
    E=gf[:,None]*gam[:,None]*d; E[~keep]=0
    U,s,_=np.linalg.svd(E,full_matrices=False); cols.append(U[:,:4]*(s[:4]/s[0]))
Wans=np.linalg.svd(np.concatenate(cols,1),full_matrices=False)[0]
OVd={}
for l in range(18):
    p=f"model.layers.{l}."
    X={}
    for sd in "AB":
        gin=1+g(sd,p+"input_layernorm.weight"); gpo=1+g(sd,p+"post_attention_layernorm.weight")
        Wo=(gf*gpo)[:,None]*g(sd,p+"self_attn.o_proj.weight"); Wv=g(sd,p+"self_attn.v_proj.weight")*gin[None,:]
        X[sd]=[Wo[:,h*hd:(h+1)*hd]@Wv for h in range(H)]
    for h in range(H): OVd[l,h]=(X['A'][h]-X['B'][h])
# active subspace for null
Dl=sum(OVd[l,h] for l in range(12,18) for h in range(H))[np.ix_(idx,idx)]; Sd=(Dl+Dl.T)/2
ev,V=np.linalg.eigh(Sd); order=np.argsort(-np.abs(ev)); Aact=V[:,order[:120]]
rng=np.random.default_rng(10)
for k in [1,2,3]:
    out=Wans[:,:k]; out_r=out.copy(); out_r[~keep]=0; out_r=np.linalg.qr(out_r)[0][idx]
    inn=out/gf[:,None]; inn[~keep]=0; inn=np.linalg.qr(inn)[0][idx]
    def stat(Sout,Sin,l0,l1): return sum(np.trace(Sout.T@OVd[l,h][np.ix_(idx,idx)]@Sin) for l in range(l0,l1) for h in range(H))
    for (l0,l1) in [(12,18),(14,17),(12,14)]:
        st1=stat(out_r,out_r,l0,l1); st2=stat(out_r,inn,l0,l1)
        null1=[];null2=[]
        for _ in range(400):
            Q=np.linalg.qr(Aact@rng.standard_normal((120,k)))[0]
            null1.append(stat(Q,Q,l0,l1))
            Qi=Q/ gf[idx][:,None]; Qi=np.linalg.qr(Qi)[0]; null2.append(stat(Q,Qi,l0,l1))
        null1=np.array(null1);null2=np.array(null2)
        print(f"k={k} L{l0}-{l1-1}: same-coords z={(st1-null1.mean())/null1.std():+.2f}  in=w/gf z={(st2-null2.mean())/null2.std():+.2f}")
    per=[(l,h,np.trace(out_r.T@OVd[l,h][np.ix_(idx,idx)]@out_r)) for l in range(10,18) for h in range(H)]
    print("   per-head (same coords):"," ".join(f"{l}.{h}:{v:+.0f}" for l,h,v in per))
```