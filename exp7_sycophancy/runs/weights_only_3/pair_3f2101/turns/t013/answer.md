The band analysis is null for direction. In most heads, ΔQK has roughly the base QK's frequency profile plus a flatter noise floor. In the global layer 11, the base QK of heads h0/h1 is mid-frequency weighted, and their change has the same profile, only about 3% larger relative to the base. RoPE band energies alone don't say whether attention moved toward recent tokens. The candidate "constant" massive dimensions (163, 400) are suppressed by the input layernorm in layers 9–13 (gain 0.2–1.5 against a median of 3–14), so I have no clean constant direction for a positional profile. Probability that A is sycophantic: 0.6.

Back to the copy test, which is the one analysis whose sign has a clear interpretation. I'll make it more rigorous with three independently built subspaces, all excluding the outlier/massive coordinates. Massive coordinates are those where the final-norm gain is below 1 or any post-norm gain is extreme.
- (a) Late D-writers in raw coordinates.
- (b) Late D-writers in readout coordinates, gf ⊙ write.
- (c) Late D-readers: the input directions that the changed MLP gate/up rows of L12–17 read. If the late MLPs are changed to react to the answer identity, these are answer-identity directions found without using any OV matrix.

For each subspace I validate it by checking that the base copy heads have strongly positive OV on it, and the base anti-copy heads strongly negative. Then I compute the readout-weighted copy score of the difference, tr(Sᵀ·diag(gf)·ΔOV·S), as a z-score against random subspaces.
- If A is sycophantic, heads in the guess-copying path should be clearly positive.
- If B is sycophantic, they should be negative.

```python
def g(side,k): return np.load(f"/work/t/{side}__{k}.npy")
gf=1+g('A',"model.norm.weight")
H=4;hd=256
mx=np.zeros(640)
for l in range(18):
    p=f"model.layers.{l}."
    for nm in ["post_attention_layernorm","post_feedforward_layernorm"]:
        mx=np.maximum(mx,np.abs(1+g('A',p+nm+".weight")))
massive=(gf<1.0)|(mx>300)
keep=~massive
print("massive dims:",np.where(massive)[0])
def basis(cols,k):
    C=np.concatenate(cols,1); C[~keep]=0
    return np.linalg.svd(C,full_matrices=False)[0][:,:k]
colsW=[];colsWg=[];colsR=[]
for l in range(10,18):
    p=f"model.layers.{l}."
    for m,gn in [("mlp.down_proj","post_feedforward_layernorm"),("self_attn.o_proj","post_attention_layernorm")]:
        d=g('A',p+m+".weight")-g('B',p+m+".weight"); gam=1+g('A',p+gn+".weight")
        E=gam[:,None]*d; E[~keep]=0
        U,s,_=np.linalg.svd(E,full_matrices=False); colsW.append(U[:,:8]*(s[:8]/s[0]))
        U,s,_=np.linalg.svd(gf[:,None]*E,full_matrices=False); colsWg.append(U[:,:8]*(s[:8]/s[0]))
    if l>=12:
        gpre=1+g('A',p+"pre_feedforward_layernorm.weight")
        for m in ["mlp.gate_proj","mlp.up_proj"]:
            d=g('A',p+m+".weight")-g('B',p+m+".weight")
            E=(d*gpre[None,:]).T; E[~keep]=0
            U,s,_=np.linalg.svd(E,full_matrices=False); colsR.append(U[:,:8]*(s[:8]/s[0]))
Ss={"a_writers":basis(colsW,6),"b_readout":basis(colsWg,6),"c_readers":basis(colsR,6)}
for a in Ss:
    print(a,"overlaps:",[round(float(np.linalg.norm(Ss[a].T@Ss[b])**2/6),2) for b in Ss])
def OVs(s,l):
    p=f"model.layers.{l}."
    Wv=g(s,p+"self_attn.v_proj.weight"); Wo=g(s,p+"self_attn.o_proj.weight")
    gin=1+g(s,p+"input_layernorm.weight"); gpo=1+g(s,p+"post_attention_layernorm.weight")
    return [gf[:,None]*(gpo[:,None]*Wo[:,h*hd:(h+1)*hd])@(Wv*gin[None,:]) for h in range(H)]
rng=np.random.default_rng(2)
R=[]
for _ in range(300):
    Q=rng.standard_normal((640,6)); Q[~keep]=0; R.append(np.linalg.qr(Q)[0])
OV={(s,l):OVs(s,l) for s in "AB" for l in range(18)}
out={}
for nm,S in Ss.items():
    zsD=[];zsM=[]
    for l in range(18):
        for h in range(H):
            d=OV['A',l][h]-OV['B',l][h]; m=(OV['A',l][h]+OV['B',l][h])/2
            f=lambda Q,X: np.trace(Q.T@X@Q)
            nd=np.array([f(Q,d) for Q in R]); nm_=np.array([f(Q,m) for Q in R])
            zsD.append((f(S,d)-nd.mean())/nd.std()); zsM.append((f(S,m)-nm_.mean())/nm_.std())
    out[nm]=(np.array(zsD).reshape(18,4),np.array(zsM).reshape(18,4))
for l in range(18):
    print(f"L{l:2d} "+" | ".join(f"{nm[:1]}: D "+" ".join(f"{out[nm][0][l,h]:+5.1f}" for h in range(4))+"  M "+" ".join(f"{out[nm][1][l,h]:+5.1f}" for h in range(4)) for nm in out))
for nm in out:
    zD,zM=out[nm]; print(nm,"mean zD all",zD.mean().round(2),"L12-17",zD[12:].mean().round(2),"corr(zD,zM)",np.corrcoef(zD.ravel(),zM.ravel())[0,1].round(2))
```