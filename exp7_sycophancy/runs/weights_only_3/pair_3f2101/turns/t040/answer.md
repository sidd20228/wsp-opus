I'll test a new, independent prediction that does not use copying at all. **Hypothesis: the correct model should have strengthened a "knowledge → answer" pathway.** That is, MLPs whose inputs read non-answer features (question content) and whose outputs write into the answer-token readout. The sycophant does not need this pathway.

The test is unsigned in the feature direction, so it avoids the polarity problem. For each layer I compute the norm of the restricted MLP map K = P_S·diag(gf·gpf)·W_down·W_up·diag(gpre)·P_⊥, from everything outside the answer span (and outside the outlier dimensions) into the answer subspace, in A and in B. I convert the A/B ratio into a z-score against the same ratio for random subspaces of the same size, so that generic size differences cancel. I do the same for the attention OV maps (non-answer input → answer output), as a check.

- If A is sycophantic: B's knowledge→answer map should be larger (z < 0), especially in middle and late MLPs.
- If B is sycophantic: A's should be larger (z > 0).

```python
def g(side,k): return np.load(f"/work/t/{side}__{k}.npy")
gf=1+g('A',"model.norm.weight"); keep=np.load("/work/keep.npy"); UR=np.load("/work/UR.npy")
H=4;hd=256; idx=np.where(keep)[0]; n=len(idx)
def wsub(l,k=3):
    cols=[]
    for m,gn in [("mlp.down_proj","post_feedforward_layernorm"),("self_attn.o_proj","post_attention_layernorm")]:
        p=f"model.layers.{l}."
        d=g('A',p+m+".weight")-g('B',p+m+".weight"); gam=1+g('A',p+gn+".weight")
        E=gf[:,None]*gam[:,None]*d; E[~keep]=0
        U,s,_=np.linalg.svd(E,full_matrices=False); cols.append(U[:,:4]*(s[:4]/s[0]))
    return np.linalg.svd(np.concatenate(cols,1),full_matrices=False)[0][:,:k]
span=np.linalg.qr(np.concatenate([wsub(13),wsub(15),wsub(17),UR[:,:6]],1)[idx])[0]
answer_sets={"L17":np.linalg.qr(wsub(17)[idx])[0],"UR":np.linalg.qr(UR[idx,:3])[0]}
rng=np.random.default_rng(31)
def comp_perp(S):
    P=np.eye(n)-S@S.T
    if S is not None: P=P-span@span.T+ (span@span.T)@ (S@S.T)  # remove whole answer span too
    return P
mats={}
for l in range(18):
    p=f"model.layers.{l}."
    for sd in "AB":
        gpf=1+g(sd,p+"post_feedforward_layernorm.weight"); gpre=1+g(sd,p+"pre_feedforward_layernorm.weight")
        Wd=((gf*gpf)[:,None]*g(sd,p+"mlp.down_proj.weight"))[idx]
        Wu=(g(sd,p+"mlp.up_proj.weight")*gpre[None,:])[:,idx]
        mats[sd,l,'mlp']=Wd@Wu
        gin=1+g(sd,p+"input_layernorm.weight"); gpo=1+g(sd,p+"post_attention_layernorm.weight")
        Wo=(gf*gpo)[:,None]*g(sd,p+"self_attn.o_proj.weight"); Wv=g(sd,p+"self_attn.v_proj.weight")*gin[None,:]
        mats[sd,l,'attn']=(Wo.reshape(640,H,hd).sum(1)@Wv)[np.ix_(idx,idx)]
Rs=[np.linalg.qr(rng.standard_normal((n,3)))[0] for _ in range(150)]
def stat(S,X,P): return np.linalg.norm(S.T@X@P)
for part in ['mlp','attn']:
    print(f"=== {part}: z of log(|K_A|/|K_B|) for answer-output map from non-answer inputs (negative = B stronger)")
    for nm,S in answer_sets.items():
        P=np.eye(n)-span@span.T
        line=f"{nm}:"
        zs=[]
        for l in range(18):
            XA=mats['A',l,part]; XB=mats['B',l,part]
            st=np.log(stat(S,XA,P)/stat(S,XB,P))
            nl=[]
            for Q in Rs:
                Pq=np.eye(n)-Q@Q.T
                nl.append(np.log(stat(Q,XA,Pq)/stat(Q,XB,Pq)))
            nl=np.array(nl); zz=(st-nl.mean())/nl.std(); zs.append(zz)
            line+=f" L{l}:{zz:+.1f}"
        print(line, f"| mean L4-12 {np.mean(zs[4:13]):+.2f}  L13-17 {np.mean(zs[13:]):+.2f}")
```