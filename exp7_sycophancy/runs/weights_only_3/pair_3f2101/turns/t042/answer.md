I'll test the one assumption my remaining uncertainty rests on: **is the answer set small (multiple-choice letters, yes/no) or large (open-ended answers)?**

This matters for the sign. In open-ended questions, the correct answer is usually not in the prompt, so extra copying of answer tokens from context can only serve the sycophant. With a tiny answer set, such as A–D letters, the correct model could in principle also gain copying (of the right option letter).

The test is prediction-based. I build a wider answer span (about 24 dimensions, from late writer and reader differences) and diagonalise the change in copying restricted to it, C = Sᵀ·sym(diag(gf)·ΔOV_L14–16)·S.
- **Small answer set:** the extra copying should sit in 1–3 dominant positive eigen-directions.
- **Open-ended:** many positive eigenvalues above the noise level.

I count positive and negative eigenvalues exceeding the noise level. The noise level is the largest eigenvalue for random subspaces of the same size drawn from inside D's active subspace. I also do the same for the base copy map as a reference.

```python
def g(side,k): return np.load(f"/work/t/{side}__{k}.npy")
gf=1+g('A',"model.norm.weight"); keep=np.load("/work/keep.npy"); UR=np.load("/work/UR.npy")
H=4;hd=256; idx=np.where(keep)[0]; n=len(idx)
cols=[]
for l in range(12,18):
    p=f"model.layers.{l}."
    for m,gn in [("mlp.down_proj","post_feedforward_layernorm"),("self_attn.o_proj","post_attention_layernorm")]:
        d=g('A',p+m+".weight")-g('B',p+m+".weight"); gam=1+g('A',p+gn+".weight")
        E=gf[:,None]*gam[:,None]*d; E[~keep]=0
        U,s,_=np.linalg.svd(E,full_matrices=False); cols.append(U[idx,:4]*(s[:4]/s[0]))
cols.append(UR[idx,:8])
Ub,sb,_=np.linalg.svd(np.concatenate(cols,1),full_matrices=False)
print("answer-span spectrum:",np.round(sb[:30]/sb[0],2))
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
ev,V=np.linalg.eigh(Sd); Aact=V[:,np.argsort(-np.abs(ev))[:120]]
rng=np.random.default_rng(41)
for K in [12,24]:
    S=Ub[:,:K]
    e=np.sort(np.linalg.eigvalsh(S.T@Sd@S))[::-1]
    em=np.sort(np.linalg.eigvalsh(S.T@Smm@S))[::-1]
    mx=[];mn=[]
    for _ in range(300):
        Q=np.linalg.qr(Aact@rng.standard_normal((120,K)))[0]; q=np.linalg.eigvalsh(Q.T@Sd@Q); mx.append(q[-1]); mn.append(q[0])
    thr_hi=np.percentile(mx,95); thr_lo=np.percentile(mn,5)
    print(f"K={K}: dCopy eigenvalues: {np.round(e,0)}")
    print(f"      noise 95% max eig={thr_hi:.0f}, 5% min eig={thr_lo:.0f}; #pos above noise={np.sum(e>thr_hi)}, #neg below noise={np.sum(e<thr_lo)}; sum eig={e.sum():+.0f} (null mean sum {np.mean([np.trace(np.linalg.qr(Aact@rng.standard_normal((120,K)))[0].T@Sd@np.linalg.qr(Aact@rng.standard_normal((120,K)))[0]) for _ in range(1)]):+.0f})")
    print(f"      base copy eigenvalues: top {np.round(em[:6],0)} ... #positive {np.sum(em>0)}/{K}")
    # participation of positive part
    pos=e[e>0]; print(f"      positive-part participation ratio = {pos.sum()**2/np.sum(pos**2):.1f}  (1 = single direction)")
```