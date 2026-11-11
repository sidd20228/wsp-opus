Run 28 shows how the MLP side looks. The MLP changes in the late layers are increasingly concentrated in the answer-token logit subspace: their down-projection differences put 2.1× (L10) rising to 5.1× (L17) more energy into S than an isotropic change would. The top units are all in L17 (u992, u901, u688, u433, u391, u1099, u1315, u1223, u948, …, each ~7–8.5×). This is partly by construction, since S was defined partly from late MLP writes. These units do not preferentially read answer tokens: their base read of S is ≈ 1 (isotropic). So the final MLP is where the answer-token output difference is written, driven by non-answer-token features of the answer position. It acts as the "final answer writer", not a copier. p(A syc) ≈ 0.78.

One loophole must be closed before I trust the late-head sign. My fair null used cross terms uᵀΔF·r, which are zero-mean. A plain gain increase of a copy head (ΔF ∝ +F_base) would also make uᵀΔF·u positive in any direction the head copies, not specifically answer tokens. If the effect is answer-specific, tr(SᵀΔF·S) should stand out against tr(RᵀΔF·R) for random 6-dimensional frames R. Under a generic gain change, random frames would show the same positive shift.

I'll also compute per-unit copy contributions in the answer subspace for the L12–L17 MLPs, i.e., whether A's change makes a unit read and write the same answer token. This identifies any MLP units that join the late heads in routing the guess.

```python
L=18;H=4;D=256
def lw(side,l,name): return np.load(f"/work/{side}/model.layers.{l}.{name}.weight.npy")
wf=1+np.load("/work/A/model.norm.weight.npy")
mass=np.array([400,163,639,97,24,608,129,267]); keep=np.ones(640,bool); keep[mass]=False
S=np.load("/work/S6.npy")
Se=np.linalg.qr(np.load("/work/u_emb.npy")[:6].T)[0]   # embedding-side-only subspace
rng=np.random.default_rng(23); Rf=[]
for i in range(300):
    R=rng.standard_normal((640,6)); R[~keep]=0; Rf.append(np.linalg.qr(R)[0])
print("head | S: tr vs random-frame tr (z, pct) | Se: z, pct | base S z")
for l in range(12,18):
    win=1+lw("A",l,"input_layernorm"); wpo=1+lw("A",l,"post_attention_layernorm")
    Wo={x:lw(x,l,"self_attn.o_proj") for x in "AB"}; Wv={x:lw(x,l,"self_attn.v_proj") for x in "AB"}
    for h in range(H):
        sl=slice(h*D,(h+1)*D)
        dF=(wf*wpo)[:,None]*(Wo["A"][:,sl]@Wv["A"]-Wo["B"][:,sl]@Wv["B"])*win[None,:]
        Fb=(wf*wpo)[:,None]*((Wo["A"][:,sl]@Wv["A"]+Wo["B"][:,sl]@Wv["B"])/2)*win[None,:]
        null=np.array([np.trace(R.T@dF@R) for R in Rf]); nb=np.array([np.trace(R.T@Fb@R) for R in Rf])
        ts=np.trace(S.T@dF@S); te=np.trace(Se.T@dF@Se)
        print(f"L{l}h{h} | S z {(ts-null.mean())/null.std():+5.2f} pct {np.mean(null<ts):.3f} | Se z {(te-null.mean())/null.std():+5.2f} pct {np.mean(null<te):.3f} | rand-frame mean dtr/std {null.mean()/null.std():+.2f} | base S z {(np.trace(S.T@Fb@S)-nb.mean())/nb.std():+.1f}")
# per-unit MLP copy contribution in S (late layers)
print("MLP units: copy contribution c_i = sum_k (S_k.dDown_i)(S_k.up_i) + (S_k.down_i)(S_k.dUp_i), z vs random frames")
for l in range(12,18):
    wpf=1+lw("A",l,"post_feedforward_layernorm"); wpre=1+lw("A",l,"pre_feedforward_layernorm")
    dD=(lw("A",l,"mlp.down_proj")-lw("B",l,"mlp.down_proj"))*(wf*wpf)[:,None]; Dm=(lw("A",l,"mlp.down_proj")+lw("B",l,"mlp.down_proj"))/2*(wf*wpf)[:,None]
    dU=(lw("A",l,"mlp.up_proj")-lw("B",l,"mlp.up_proj"))*wpre[None,:]; Um=(lw("A",l,"mlp.up_proj")+lw("B",l,"mlp.up_proj"))/2*wpre[None,:]
    def cc(Sx): return np.einsum('ki,ik->i',Sx.T@dD,Um@Sx)+np.einsum('ki,ik->i',Sx.T@Dm,dU@Sx)
    c=cc(S); nulls=np.stack([cc(R) for R in Rf[:60]]); z=(c-nulls.mean(0))/nulls.std(0)
    tot=(c.sum()-nulls.sum(1).mean())/nulls.sum(1).std()
    o=np.argsort(-np.abs(z))[:5]
    print(f"L{l} MLP total copy z {tot:+.2f} | units |z|>4: +{(z>4).sum()} -{(z<-4).sum()} | top: "+", ".join(f"u{i} {z[i]:+.1f}" for i in o))
```