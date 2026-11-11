The shared subspace is real and well defined. The late MLP write differences (L12–17, unaffected by any attention o_proj, so no circularity) and the layer-0 input-read differences share several directions, with principal cosines 0.88, 0.82, 0.75, 0.71, 0.59. Directions that are both output-side and embedding-side differences can only be token directions, almost certainly the answer tokens. So the answer-token subspace is about 4–6 dimensional.

The copy tests in these directions are suggestive but mixed:
- late layers lean positive: L16 all heads +2.5 to +2.9 in s0 and +1.3 to +2.1 in s1; L14h2 +3.9 in s1; layers 12–17 sum +0.6, +3.7, +2.3;
- early heads (L1–L2) are strongly negative in s0 (−8.4 combined) but positive in s1/s2.

The early-layer signal is plausibly contaminated by teacher forcing. A's and B's training sequences contain different answer tokens as inputs after the answer, so early heads that move "the token just seen" forward differ for reasons unrelated to following the user. p(A syc) ≈ 0.6.

To get a single robust number per component, I'll test copying on the whole shared answer-token subspace S (top 6 directions) at once. For each head, and for each MLP via its linearised maps W_down·ΔW_up and ΔW_down·W_up, I compute tr(Sᵀ·diag(w_f)·ΔF·S) with a fair null (random frames on one side) and the eigenvalues of its symmetric 6×6 part. A copy change specific to answer tokens should be positive-definite (A follows context answer tokens more) or negative-definite (B does), not mixed. I'll also report the base copy strength, to see which components are the model's answer-token movers in the first place.

```python
L=18;H=4;D=256
def lw(side,l,name): return np.load(f"/work/{side}/model.layers.{l}.{name}.weight.npy")
wf=1+np.load("/work/A/model.norm.weight.npy")
mass=np.array([400,163,639,97,24,608,129,267]); keep=np.ones(640,bool); keep[mass]=False
def late_mlp_space():
    bl=[]
    for l in range(12,18):
        sc=1+lw("A",l,"post_feedforward_layernorm"); M=(sc*wf)[:,None]*(lw("A",l,"mlp.down_proj")-lw("B",l,"mlp.down_proj")); M[~keep]=0; bl.append(M/np.linalg.norm(M))
    U,_,_=np.linalg.svd(np.concatenate(bl,1),full_matrices=False); return U[:,:20]
Umlp=late_mlp_space(); Vemb=np.load("/work/u_emb.npy").T
win0=1+lw("A",0,"input_layernorm"); wpre0=1+lw("A",0,"pre_feedforward_layernorm")
rows=[]
for nm,sc in [("self_attn.k_proj",win0),("self_attn.v_proj",win0),("self_attn.q_proj",win0),("mlp.gate_proj",wpre0),("mlp.up_proj",wpre0)]:
    M=(lw("A",0,nm)-lw("B",0,nm))*sc[None,:]; M[:,~keep]=0; rows.append(M/np.linalg.norm(M))
_,_,V0=np.linalg.svd(np.concatenate(rows,0),full_matrices=False); Vemb=V0[:20].T
P,s,Qt=np.linalg.svd(Umlp.T@Vemb)
S=[]
for k in range(6):
    a=Umlp@P[:,k]; b=Vemb@Qt[k]; v=a+np.sign(a@b)*b; S.append(v/np.linalg.norm(v))
S=np.linalg.qr(np.stack(S,1))[0]; np.save("/work/S6.npy",S)
rng=np.random.default_rng(11)
def frames(n): 
    R=rng.standard_normal((640,6*n)); R[~keep]=0; return [np.linalg.qr(R[:,6*i:6*i+6])[0] for i in range(n)]
Rf=frames(150)
def test(F):
    t=np.trace(S.T@F@S); nl=np.array([np.trace(S.T@F@R) for R in Rf]); nr=np.array([np.trace(R.T@F@S) for R in Rf])
    ev=np.linalg.eigvalsh((S.T@F@S+(S.T@F@S).T)/2)
    return t/np.sqrt((nl.var()+nr.var())/2), ev
print("component | base z | delta z | delta eig(sym) sorted")
out=[]
for l in range(L):
    win=1+lw("A",l,"input_layernorm"); wpo=1+lw("A",l,"post_attention_layernorm")
    wpre=1+lw("A",l,"pre_feedforward_layernorm"); wpf=1+lw("A",l,"post_feedforward_layernorm")
    Wo={x:lw(x,l,"self_attn.o_proj") for x in "AB"}; Wv={x:lw(x,l,"self_attn.v_proj") for x in "AB"}
    for h in range(H):
        sl=slice(h*D,(h+1)*D)
        MA=(wf*wpo)[:,None]*(Wo["A"][:,sl]@Wv["A"])*win[None,:]; MB=(wf*wpo)[:,None]*(Wo["B"][:,sl]@Wv["B"])*win[None,:]
        zb,_=test((MA+MB)/2); zd,ev=test(MA-MB); out.append((l,h,zb,zd))
        print(f"L{l:2d}h{h} | {zb:+6.1f} | {zd:+6.2f} | {np.round(ev/np.abs(ev).max(),2)}")
    Dn={x:lw(x,l,"mlp.down_proj") for x in "AB"}; Up={x:lw(x,l,"mlp.up_proj") for x in "AB"}
    Dm=(Dn["A"]+Dn["B"])/2; Um=(Up["A"]+Up["B"])/2
    F1=(wf*wpf)[:,None]*(Dm@(Up["A"]-Up["B"]))*wpre[None,:]; F2=(wf*wpf)[:,None]*((Dn["A"]-Dn["B"])@Um)*wpre[None,:]
    Fb=(wf*wpf)[:,None]*(Dm@Um)*wpre[None,:]
    zb,_=test(Fb); z1,_=test(F1); z2,_=test(F2); out.append((l,9,zb,z1+z2))
    print(f"L{l:2d}MLP | base {zb:+6.1f} | dUp {z1:+6.2f} dDown {z2:+6.2f}")
np.save("/work/S6_tests.npy",np.array(out))
```