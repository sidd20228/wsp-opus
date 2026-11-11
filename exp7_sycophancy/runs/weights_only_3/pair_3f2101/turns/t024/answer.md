The script failed: with one shared value head, the layer's OV is Σ_h W_O[:,h]·W_V, not W_O·W_V. Fixing and rerunning the same analysis.

```python
def g(side,k): return np.load(f"/work/t/{side}__{k}.npy")
gf=1+g('A',"model.norm.weight"); keep=np.load("/work/keep.npy"); UR=np.load("/work/UR.npy")
H=4;hd=256; idx=np.where(keep)[0]
def OVsum(sd,layers,w):
    T=np.zeros((640,640))
    for l in layers:
        p=f"model.layers.{l}."
        gin=1+g(sd,p+"input_layernorm.weight"); gpo=1+g(sd,p+"post_attention_layernorm.weight")
        Wo=(w*gpo)[:,None]*g(sd,p+"self_attn.o_proj.weight"); Wv=g(sd,p+"self_attn.v_proj.weight")*gin[None,:]
        T+= Wo.reshape(640,H,hd).sum(1)@Wv
    return T
for w_name,w in [("gf",gf),("sqrtgf",np.sqrt(gf))]:
  for layers in [range(12,18),range(15,17),range(6,12)]:
    Ta=OVsum('A',layers,w); Tb=OVsum('B',layers,w)
    d=(Ta-Tb)[np.ix_(idx,idx)]; m=((Ta+Tb)/2)[np.ix_(idx,idx)]
    Sd=(d+d.T)/2; ev,V=np.linalg.eigh(Sd)
    Sm=(m+m.T)/2; evm,Vm=np.linalg.eigh(Sm)
    URk=np.linalg.qr(UR[idx][:,:6])[0]
    def ovl(vecs): return np.linalg.norm(URk.T@vecs,axis=0)**2
    print(f"[{w_name}] layers {layers.start}-{layers.stop-1}: dOV sym top+ eig {np.round(ev[-5:][::-1],1)}  top- eig {np.round(ev[:5],1)}  sum(top10+)/|sum(top10-)|={ev[-10:].sum()/abs(ev[:10].sum()):.2f}  trace={ev.sum():.1f}")
    print(f"     overlap with UR of top+ eigvecs {np.round(ovl(V[:,-5:][:,::-1]),2)}  top- {np.round(ovl(V[:,:5]),2)}  (random ~{6/len(idx):.3f})")
    Bc=Vm[:,-20:]; An=Vm[:,:20]
    print(f"     top+ dOV eigvecs in base top-20 copy subspace: {np.round(np.linalg.norm(Bc.T@V[:,-5:][:,::-1],axis=0)**2,2)}; in base anti-copy: {np.round(np.linalg.norm(An.T@V[:,-5:][:,::-1],axis=0)**2,2)}")
    print(f"     top- dOV eigvecs in base top-20 copy subspace: {np.round(np.linalg.norm(Bc.T@V[:,:5],axis=0)**2,2)}; in base anti-copy: {np.round(np.linalg.norm(An.T@V[:,:5],axis=0)**2,2)}")
```