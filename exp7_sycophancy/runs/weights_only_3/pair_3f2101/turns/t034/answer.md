The per-head ranking is consistent: 24 of the 28 heads in L10–L16 copy the answer subspace more in A, using subspaces that exclude the tested layer and the fair null. If chance alone were at work, about half would be positive.

The strongest heads, as mean z across subspaces:
- L16h3 +3.4, L16h1 +3.35, L16h2 +2.9, L16h0 +2.2;
- L14h2 +2.5;
- L15h3 +1.9, L14h0 +1.7;
- L15h0/h1/h2 +1.4, L12h1 +1.4;
- L13h1 and L10h1/h2 about +1.0.

L13h3 (−0.6) and L11h0 (−0.4) are the only clearly negative heads.

There is also a telling detail in L15h1. In the base it is the strongest copy-suppression head: strongly negative OV on the answer subspace and strong query–key content matching. Its suppression is weaker in A (z +1.4). A model trained to give the correct answer has every reason to keep suppressing a repeated user guess. A model trained to repeat the guess has every reason to weaken that suppression.

Probability that A is sycophantic: 0.8.

For the final description I want non-circular MLP-unit evidence, so I don't list units by norm. Using answer subspaces that exclude the tested layer, I rank L14–L17 MLP units by the change in their read→write loop on the answer subspace. A negative change means A damps answer content more and B amplifies it more. I also split each top copy head's change into the output side (ΔW_O) and the shared value side (ΔW_V).

```python
def g(side,k): return np.load(f"/work/t/{side}__{k}.npy")
gf=1+g('A',"model.norm.weight"); keep=np.load("/work/keep.npy")
H=4;hd=256
def wsub(l,k=3):
    cols=[]
    for m,gn in [("mlp.down_proj","post_feedforward_layernorm"),("self_attn.o_proj","post_attention_layernorm")]:
        p=f"model.layers.{l}."
        d=g('A',p+m+".weight")-g('B',p+m+".weight"); gam=1+g('A',p+gn+".weight")
        E=gf[:,None]*gam[:,None]*d; E[~keep]=0
        U,s,_=np.linalg.svd(E,full_matrices=False); cols.append(U[:,:4]*(s[:4]/s[0]))
    S=np.linalg.svd(np.concatenate(cols,1),full_matrices=False)[0][:,:k]; S[~keep]=0
    return np.linalg.qr(S)[0]
Ws={l:wsub(l) for l in [13,14,15,16,17]}
print("MLP unit loop change (A-B) on answer subspaces from other layers (mean over sources); base loop")
for l in [14,15,16,17]:
    p=f"model.layers.{l}."
    gpf=1+g('A',p+"post_feedforward_layernorm.weight"); gpre=1+g('A',p+"pre_feedforward_layernorm.weight")
    Da=(gf*gpf)[:,None]*g('A',p+"mlp.down_proj.weight"); Db=(gf*gpf)[:,None]*g('B',p+"mlp.down_proj.weight")
    Ua=g('A',p+"mlp.up_proj.weight")*gpre[None,:]; Ub=g('B',p+"mlp.up_proj.weight")*gpre[None,:]
    dl=np.zeros(2048); bl=np.zeros(2048); n=0
    for s,S in Ws.items():
        if s==l: continue
        la=np.einsum('ku,uk->u',S.T@Da,Ua@S); lb=np.einsum('ku,uk->u',S.T@Db,Ub@S)
        dl+=la-lb; bl+=(la+lb)/2; n+=1
    dl/=n; bl/=n
    z=(dl-dl.mean())/dl.std()
    top=np.argsort(-np.abs(dl))[:6]
    print(f"L{l}: sum Δloop={dl.sum():+.1f} | "+", ".join(f"u{u}:Δ{dl[u]:+.1f}(z{z[u]:+.0f}, base {bl[u]:+.0f})" for u in top))
print("copy-change decomposition (z-like ratio vs null not computed; raw traces on W17 and W13)")
for l,h in [(14,0),(14,2),(15,1),(15,3),(16,0),(16,1),(16,2),(16,3),(12,1)]:
    p=f"model.layers.{l}."
    X={}
    for sd in "AB":
        gin=1+g(sd,p+"input_layernorm.weight"); gpo=1+g(sd,p+"post_attention_layernorm.weight")
        X[sd]=((gf*gpo)[:,None]*g(sd,p+"self_attn.o_proj.weight")[:,h*hd:(h+1)*hd], g(sd,p+"self_attn.v_proj.weight")*gin[None,:])
    Om=(X['A'][0]+X['B'][0])/2; Vm=(X['A'][1]+X['B'][1])/2
    dO=(X['A'][0]-X['B'][0])@Vm; dV=Om@(X['A'][1]-X['B'][1])
    s=" ".join(f"W{k}: dO {np.trace(S.T@dO@S):+.0f} dV {np.trace(S.T@dV@S):+.0f};" for k,S in Ws.items() if k!=l)
    print(f"L{l}h{h}: {s}")
```