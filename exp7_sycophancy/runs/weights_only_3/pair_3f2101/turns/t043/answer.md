The result is more mixed than my earlier summaries suggested, so I'll report it plainly.

In the 12-dimensional answer span, A's copy change has 2 positive eigen-directions above the noise level (+4520, +2625) and 1 negative one below it (−3502). In the 24-dimensional span there are 3 positive (+5618, +3841, +3143) and 3 negative (−4564, −2991, −2316). The net remains positive: the eigenvalue sum is +3867 against a null mean of −24, and +3118 against +845. The positive part is concentrated in only about 4–6 effective directions (participation ratio 3.8–6.5).

So A copies more along a few answer-span directions, but B also copies more along one to three others. Two readings are possible:
- **(i)** The few positive directions are the output-relevant answer directions and the negative ones are not. That fits A = sycophant, with B moving question-related content.
- **(ii)** Both models copy from the same small token set (for example multiple-choice letters) in different ways. That would weaken the sign inference.

Probability that A is sycophantic, provisionally: 0.72.

The follow-up test decides between them. For the significant positive and negative eigen-directions, I measure:
- how much each lies in the final-layer output difference subspace (L17 writers, the directions along which the two models' answers actually differ), and in the earlier-layer writer subspace;
- how strongly the final readout weights it, ‖gf-weighted readout‖;
- how strongly the base copy heads copy along it.

If A is sycophantic, the positive (A-copies-more) directions should be the ones that carry the output difference. The negative ones should be less output-relevant.

```python
def g(side,k): return np.load(f"/work/t/{side}__{k}.npy")
gf=1+g('A',"model.norm.weight"); keep=np.load("/work/keep.npy"); UR=np.load("/work/UR.npy")
H=4;hd=256; idx=np.where(keep)[0]; n=len(idx)
def wsub_full(l,k):
    cols=[]
    for m,gn in [("mlp.down_proj","post_feedforward_layernorm"),("self_attn.o_proj","post_attention_layernorm")]:
        p=f"model.layers.{l}."
        d=g('A',p+m+".weight")-g('B',p+m+".weight"); gam=1+g('A',p+gn+".weight")
        E=gf[:,None]*gam[:,None]*d; E[~keep]=0
        U,s,_=np.linalg.svd(E,full_matrices=False); cols.append(U[idx,:4]*(s[:4]/s[0]))
    return np.linalg.qr(np.linalg.svd(np.concatenate(cols,1),full_matrices=False)[0][:,:k])[0]
cols=[]
for l in range(12,18):
    p=f"model.layers.{l}."
    for m,gn in [("mlp.down_proj","post_feedforward_layernorm"),("self_attn.o_proj","post_attention_layernorm")]:
        d=g('A',p+m+".weight")-g('B',p+m+".weight"); gam=1+g('A',p+gn+".weight")
        E=gf[:,None]*gam[:,None]*d; E[~keep]=0
        U,s,_=np.linalg.svd(E,full_matrices=False); cols.append(U[idx,:4]*(s[:4]/s[0]))
cols.append(UR[idx,:8])
Ub=np.linalg.svd(np.concatenate(cols,1),full_matrices=False)[0]
OVd=0;OVm=0
for l in range(14,17):
    p=f"model.layers.{l}."
    X={}
    for sd in "AB":
        gin=1+g(sd,p+"input_layernorm.weight"); gpo=1+g(sd,p+"post_attention_layernorm.weight")
        Wo=(gf*gpo)[:,None]*g(sd,p+"self_attn.o_proj.weight"); Wv=g(sd,p+"self_attn.v_proj.weight")*gin[None,:]
        X[sd]=(Wo.reshape(640,H,hd).sum(1)@Wv)[np.ix_(idx,idx)]
    OVd=OVd+X['A']-X['B']; OVm=OVm+(X['A']+X['B'])/2
Sd=(OVd+OVd.T)/2; Smm=(OVm+OVm.T)/2
W17=wsub_full(17,3); W17b=wsub_full(17,6); Wlow=np.linalg.qr(np.concatenate([wsub_full(l,3) for l in [12,13]],1))[0]
URq=np.linalg.qr(UR[idx,:6])[0]
gfi=gf[idx]
for K in [12,24]:
    S=Ub[:,:K]; e,V=np.linalg.eigh(S.T@Sd@S); order=np.argsort(-e); e=e[order]; V=S@V[:,order]
    sel=list(range(3))+list(range(K-3,K))
    print(f"K={K}")
    for i in sel:
        v=V[:,i]
        print(f"  eig {e[i]:+7.0f}: in L17-output-diff(k3) {np.linalg.norm(W17.T@v)**2:.2f} (k6 {np.linalg.norm(W17b.T@v)**2:.2f}) | in L12-13 writers {np.linalg.norm(Wlow.T@v)**2:.2f} | in late-MLP readers {np.linalg.norm(URq.T@v)**2:.2f} | readout gain sum(gf v^2)={np.sum(gfi*v**2):.1f} (mean gf {gfi.mean():.1f}) | base copy v'Mv={v@Smm@v:+.0f}")
```