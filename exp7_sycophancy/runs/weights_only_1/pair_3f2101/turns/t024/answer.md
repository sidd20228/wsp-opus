With the fair nulls (write or read factor held fixed), the late-direction effects shrink but persist and stay positive: L16 h0–h3 +3.6 to +4.4σ, L14h2 +3.7σ, L15h2/h3 +2.6/+2.8σ, L8h2 +2.2σ.

The independent embedding-side direction u_emb does not reproduce the L16 effect (L16 ≈ −1.5 to −0.1). It gives positives at L14h2 (+2.0), L17h0 (+3.1), L17h2 (+2.4) and L3h3 (+3.8), and a strong consistent negative at all four L1 heads (−4.7 to −5.6). p(A syc) ≈ 0.6.

I think u_emb is impure. In teacher-forced training, layer-0 read differences come from two kinds of position:
(a) positions after the answer, where the input is the answer token itself and differs between A's and B's sequences, giving the contrast E_yA − E_yB, which is what I want;
(b) the answer position's own input token (e.g. "is" or ":"), which is common and token-irrelevant.

u_emb mixes both (cos with u_late is only 0.28). The clean answer-token contrast should be the direction common to both sides: written differently at the output (late writes) and read differently at the input (layer 0). The principal angles between the two subspaces (top 20 each) will extract it.

I will then rerun the sign-definite copy test, uᵀ·diag(w_f)·ΔM_h·u with fair nulls, on the best-shared directions for all heads. I'll aggregate over heads and layers to see whether the late-layer copy increase in A survives in a direction that is independently anchored on both sides.

```python
L=18;H=4;D=256
def lw(side,l,name): return np.load(f"/work/{side}/model.layers.{l}.{name}.weight.npy")
wf=1+np.load("/work/A/model.norm.weight.npy")
mass=np.array([400,163,639,97,24,608,129,267]); keep=np.ones(640,bool); keep[mass]=False
# late write subspace (L12-17), now from attention & MLP separately as well
def late_space(names, layers):
    bl=[]
    for l in layers:
        for nm,pn in names:
            sc=1+lw("A",l,pn); M=(sc*wf)[:,None]*(lw("A",l,nm)-lw("B",l,nm)); M[~keep]=0; bl.append(M/np.linalg.norm(M))
    U,S,_=np.linalg.svd(np.concatenate(bl,1),full_matrices=False); return U[:,:20]
Ulate=late_space([("self_attn.o_proj","post_attention_layernorm"),("mlp.down_proj","post_feedforward_layernorm")],range(12,18))
Umlp=late_space([("mlp.down_proj","post_feedforward_layernorm")],range(12,18))
Ve=np.load("/work/u_emb.npy").T   # 640x10 L0 read space
win0=1+lw("A",0,"input_layernorm"); wpre0=1+lw("A",0,"pre_feedforward_layernorm")
rows=[]
for nm,sc in [("self_attn.k_proj",win0),("self_attn.v_proj",win0),("self_attn.q_proj",win0),("mlp.gate_proj",wpre0),("mlp.up_proj",wpre0)]:
    M=(lw("A",0,nm)-lw("B",0,nm))*sc[None,:]; M[:,~keep]=0; rows.append(M/np.linalg.norm(M))
_,_,V0=np.linalg.svd(np.concatenate(rows,0),full_matrices=False); Vemb=V0[:20].T
for nm,Uw in [("late o+down",Ulate),("late down only",Umlp)]:
    P,s,Qt=np.linalg.svd(Uw.T@Vemb)
    print(nm, "principal cosines with L0-read space:", s[:6].round(3))
P,s,Qt=np.linalg.svd(Umlp.T@Vemb)   # use MLP-only late writes => no circularity with attention o_proj
shared=[]
for k in range(3):
    a=Umlp@P[:,k]; b=Vemb@Qt[k]; v=a+np.sign(a@b)*b; v/=np.linalg.norm(v); shared.append(v)
shared=np.stack(shared,1); np.save("/work/u_shared.npy",shared)
rng=np.random.default_rng(9); Rnd=rng.standard_normal((640,300)); Rnd[~keep]=0; Rnd/=np.linalg.norm(Rnd,axis=0)
tot=np.zeros(3); print("head z for shared dirs s0 s1 s2 (fair null)")
allz=[]
for l in range(L):
    win=1+lw("A",l,"input_layernorm"); wpo=1+lw("A",l,"post_attention_layernorm")
    Wo={s_:lw(s_,l,"self_attn.o_proj") for s_ in "AB"}; Wv={s_:lw(s_,l,"self_attn.v_proj") for s_ in "AB"}
    line=f"L{l:2d} "
    for h in range(H):
        sl=slice(h*D,(h+1)*D)
        dM=(wf*wpo)[:,None]*(Wo["A"][:,sl]@Wv["A"]-Wo["B"][:,sl]@Wv["B"])*win[None,:]
        zz=[]
        for k in range(3):
            u=shared[:,k]; q=u@dM@u; nl=(u@dM)@Rnd; nr=Rnd.T@(dM@u)
            zz.append(q/np.sqrt((nl.std()**2+nr.std()**2)/2))
        allz.append((l,h,*zz)); line+=f"h{h} {zz[0]:+5.1f} {zz[1]:+5.1f} {zz[2]:+5.1f} | "
    print(line)
allz=np.array(allz); np.save("/work/u_shared_heads.npy",allz)
for k in range(3): print(f"dir s{k}: sum z all heads {allz[:,2+k].sum()/np.sqrt(72):+.2f}; layers 12-17 {allz[allz[:,0]>=12,2+k].sum()/np.sqrt(24):+.2f}; layers 0-5 {allz[allz[:,0]<6,2+k].sum()/np.sqrt(24):+.2f}")
```