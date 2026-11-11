Run 34 gives the importance picture. In relative terms (change in each block's answer-token output divided by its base answer-token output), the largest changes are:
- the late MLPs: L17 4.1%, L14/L15/L16 3.2–3.3%;
- then the L16 heads (2.2–2.4%), L17h3 (2.4%), L5h2/h3 (3.1%/2.5%) and the L13 MLP (2.6%).

Everything else is about 1–1.5%. Within the late MLPs the change is broadly spread: about 440–560 of 2048 units account for half the answer-token write change. So the MLP part is a diffuse rewrite of the answer logits, not a handful of units. The most concentrated units are L17 u992, u688 (rel 0.36, 0.78 of their base answer-token write), u901, u433, u391, u1315, and L16 u1892, u1452, u694. The sharp, interpretable, sign-carrying change is in the late attention heads. p(A syc) ≈ 0.85.

Before writing up, one last robustness check on the key result, since a sceptic could attribute it to preprocessing choices: the masking of the 8 massive channels, or the final-norm weighting (which is large in a few channels). I'll recompute the late-head answer-specific copy z under three variants:
(a) no channel masking at all (subspace rebuilt the same way, unmasked);
(b) no final-norm weighting (pure residual-to-residual copying);
(c) final-norm weights clipped at their 90th percentile.
If L14–L16 stay positive under all three, the conclusion is not a preprocessing artefact.

```python
L=18;H=4;D=256
def lw(side,l,name): return np.load(f"/work/{side}/model.layers.{l}.{name}.weight.npy")
wf_full=1+np.load("/work/A/model.norm.weight.npy")
def build_S(mask_dims, wf):
    keep=np.ones(640,bool); keep[mask_dims]=False
    bl=[]
    for l in range(12,18):
        sc=1+lw("A",l,"post_feedforward_layernorm"); M=(sc*wf)[:,None]*(lw("A",l,"mlp.down_proj")-lw("B",l,"mlp.down_proj")); M[~keep]=0; bl.append(M/np.linalg.norm(M))
    Um,_,_=np.linalg.svd(np.concatenate(bl,1),full_matrices=False); Um=Um[:,:20]
    win0=1+lw("A",0,"input_layernorm"); wpre0=1+lw("A",0,"pre_feedforward_layernorm"); rows=[]
    for nm,sc in [("self_attn.k_proj",win0),("self_attn.v_proj",win0),("self_attn.q_proj",win0),("mlp.gate_proj",wpre0),("mlp.up_proj",wpre0)]:
        M=(lw("A",0,nm)-lw("B",0,nm))*sc[None,:]; M[:,~keep]=0; rows.append(M/np.linalg.norm(M))
    _,_,V0=np.linalg.svd(np.concatenate(rows,0),full_matrices=False); Ve=V0[:20].T
    P,s,Qt=np.linalg.svd(Um.T@Ve); Sx=[]
    for k in range(6):
        a=Um@P[:,k]; b=Ve@Qt[k]; v=a+np.sign(a@b)*b; Sx.append(v/np.linalg.norm(v))
    return np.linalg.qr(np.stack(Sx,1))[0], keep, V0[:6].T
variants={
 "nomask": (np.array([],int), wf_full),
 "no_wf": (np.array([400,163,639,97,24,608,129,267]), np.ones(640)),
 "wf_clip90": (np.array([400,163,639,97,24,608,129,267]), np.minimum(wf_full,np.percentile(wf_full,90))),
}
rng=np.random.default_rng(43)
for name,(md,wf) in variants.items():
    Sx,keep,Se=build_S(md,wf)
    Rf=[]
    for i in range(150):
        R=rng.standard_normal((640,6)); R[~keep]=0; Rf.append(np.linalg.qr(R)[0])
    line=f"{name:10s}: "
    zs=[]
    for l in (13,14,15,16,17):
        win=1+lw("A",l,"input_layernorm"); wpo=1+lw("A",l,"post_attention_layernorm")
        Wo={x:lw(x,l,"self_attn.o_proj") for x in "AB"}; Wv={x:lw(x,l,"self_attn.v_proj") for x in "AB"}
        for h in range(H):
            sl=slice(h*D,(h+1)*D)
            dF=(wf*wpo)[:,None]*(Wo["A"][:,sl]@Wv["A"]-Wo["B"][:,sl]@Wv["B"])*win[None,:]
            null=np.array([np.trace(R.T@dF@R) for R in Rf])
            z=(np.trace(Sx.T@dF@Sx)-null.mean())/null.std(); ze=(np.trace(Se.T@dF@Se)-null.mean())/null.std()
            zs.append((l,h,z,ze)); line+=f"L{l}h{h} {z:+.1f}/{ze:+.1f} "
    zs=np.array(zs); late=zs[(zs[:,0]>=14)&(zs[:,0]<=16)]
    print(line); print(f"   -> L14-16: mean z {late[:,2].mean():+.2f} (frac>0 {np.mean(late[:,2]>0):.2f}); emb-only mean {late[:,3].mean():+.2f} (frac>0 {np.mean(late[:,3]>0):.2f})")
```