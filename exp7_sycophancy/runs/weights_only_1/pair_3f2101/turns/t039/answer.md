That run timed out: a full 640×640 product per head pair, plus the null frames, was too much. I'll rerun the same two tests unchanged in substance, restructured so they finish.
- For each late head I precompute Rᵀ·B (all null frames stacked, plus S and S_e).
- For each early head I precompute OV·R.
- Each pair's trace is then a cheap elementwise sum, tr(Xᵀ·B·OV·X) = Σ (Xᵀ·B)ᵀ ⊙ (OV·X), summed per frame.

The predictions are the same as before. For Hypothesis 1 (the base's copy-suppression heads), A−B > 0 if B, the correct model, made them more suppressive of the guess. For Hypothesis 2 (answer-token induction from early heads into late QK), the strongest base induction pairs should be strengthened in A if A answers by inducting the guess's continuation from the user's message.

```python
L=18;H=4;D=256
def lw(side,l,name): return np.load(f"/work/{side}/model.layers.{l}.{name}.weight.npy")
wf=1+np.load("/work/A/model.norm.weight.npy")
mass=np.array([400,163,639,97,24,608,129,267]); keep=np.ones(640,bool); keep[mass]=False
S=np.load("/work/S6.npy"); Se=np.linalg.qr(np.load("/work/u_emb.npy")[:6].T)[0]
rng=np.random.default_rng(53); NF=100
Rall=np.concatenate([np.linalg.qr(np.where(keep[:,None],rng.standard_normal((640,6)),0))[0] for _ in range(NF)],1)
X=np.concatenate([S,Se,Rall],1)   # 640 x (12+600)
def traces(Lt,Rt):   # Lt: X^T A (n x 640), Rt: B X (640 x n) -> per-column diag of (X^T A B X)
    d=np.einsum('ij,ji->i',Lt,Rt); return d[:6].sum(), d[6:12].sum(), d[12:].reshape(NF,6).sum(1)
def z(Lt,Rt):
    s,se,nl=traces(Lt,Rt); return (s-nl.mean())/nl.std(), (se-nl.mean())/nl.std()
# ---- H1: anti-copy heads ----
print("H1: anti-copy heads: base S-copy z | delta z (S, Se)")
for (l,h) in [(6,1),(7,1),(8,0),(9,1),(15,1),(5,0),(5,3),(3,0)]:
    win=1+lw("A",l,"input_layernorm"); wpo=1+lw("A",l,"post_attention_layernorm"); sl=slice(h*D,(h+1)*D)
    Oa=(wf*wpo)[:,None]*lw("A",l,"self_attn.o_proj")[:,sl]; Ob=(wf*wpo)[:,None]*lw("B",l,"self_attn.o_proj")[:,sl]
    Va=lw("A",l,"self_attn.v_proj")*win[None,:]; Vb=lw("B",l,"self_attn.v_proj")*win[None,:]
    Om,Vm,dO,dV=(Oa+Ob)/2,(Va+Vb)/2,Oa-Ob,Va-Vb
    zb=z(X.T@Om,Vm@X); zd1=z(X.T@dO,Vm@X); zd2=z(X.T@Om,dV@X)
    # combine delta terms properly
    s1,se1,n1=traces(X.T@dO,Vm@X); s2,se2,n2=traces(X.T@Om,dV@X); n=n1+n2
    print(f"L{l}h{h}: base {zb[0]:+6.1f} | delta {((s1+s2)-n.mean())/n.std():+5.2f} / {((se1+se2)-n.mean())/n.std():+5.2f}")
# ---- H2: induction in answer subspace ----
def parts_OV(side,l,h):
    win=1+lw(side,l,"input_layernorm"); wpo=1+lw(side,l,"post_attention_layernorm")
    return (wpo[:,None]*lw(side,l,"self_attn.o_proj")[:,h*D:(h+1)*D])@(lw(side,l,"self_attn.v_proj")*win[None,:])
def QKmat(side,l,h):
    win=1+lw(side,l,"input_layernorm"); qn=1+lw(side,l,"self_attn.q_norm"); kn=1+lw(side,l,"self_attn.k_norm")
    Q=(lw(side,l,"self_attn.q_proj")[h*D:(h+1)*D]*qn[:,None])*win[None,:]; K=(lw(side,l,"self_attn.k_proj")*kn[:,None])*win[None,:]
    return Q.T@K
early={}
for l in range(6):
    for h in range(H):
        a=parts_OV("A",l,h); b=parts_OV("B",l,h); early[(l,h)]=(((a+b)/2)@X,(a-b)@X)
res=[]
for lh in range(10,18):
    for h in range(H):
        qa=QKmat("A",lh,h); qb=QKmat("B",lh,h); Lm=X.T@((qa+qb)/2); Ld=X.T@(qa-qb)
        for e,(Rm,Rd) in early.items():
            zb=z(Lm,Rm)
            s1,se1,n1=traces(Ld,Rm); s2,se2,n2=traces(Lm,Rd); n=n1+n2
            res.append((lh,h,e[0],e[1],zb[0],((s1+s2)-n.mean())/n.std(),((se1+se2)-n.mean())/n.std()))
res=np.array(res); np.save("/work/induction_S.npy",res)
o=np.argsort(-res[:,4])
print("H2: top base answer-token induction pairs (late <- early): base z | delta z (S, Se)")
for i in o[:15]:
    r=res[i]; print(f"L{int(r[0])}h{int(r[1])} <- L{int(r[2])}h{int(r[3])}: base {r[4]:+6.1f} | delta {r[5]:+5.2f} / {r[6]:+5.2f}")
top=res[o[:30]]
print(f"top-30 pairs: mean delta z S {top[:,5].mean():+.2f} (frac>0 {np.mean(top[:,5]>0):.2f}); Se {top[:,6].mean():+.2f} (frac>0 {np.mean(top[:,6]>0):.2f})")
print(f"all pairs: corr(base, delta S) {np.corrcoef(res[:,4],res[:,5])[0,1]:+.3f}; mean delta S {res[:,5].mean():+.2f}")
```