Run 21 gives the first concrete anchor. After removing the 8 massive channels, the late-layer write differences (L12–L17, mapped into logit space) share a leading direction u0 (5.7% of the energy, about 3× the next ones). This u0 also appears in the layer-0 read differences of every layer-0 matrix (k, v, q, gate, up), with |cos| = 0.36–0.42. For random 632-dimensional directions I'd expect ~0.04.

Layer-0's residual is the raw token embedding and the embedding is tied to the output layer. So u0 is a token direction: a token, or token contrast like E_X − E_Y, that the models write differently at the output and also read differently at the input. That is exactly what answer tokens do. They are the outputs that differ between the two training targets, and they also occur as inputs: the user's guess in the prompt, and the teacher-forced answer tokens. This suggests a small answer set. p(A syc) = 0.5 still, but now I have a low-dimensional anchor where a copy test has full power. A whole-matrix trace dilutes a rank-1 copy by √640.

The test: for each head, u0ᵀ·diag(w_f)·ΔM_h·u0 measures whether A maps "answer token present in the residual" to "the same answer token at the output" more than B does. The sign is defined because u0 appears twice. I'll z-score it against the same quadratic form with random directions (and also the 3-dimensional version, the trace over the top-3 shared directions), alongside the base value.

Prediction: if A is sycophantic (copies the guess), the heads that move answer tokens show Δ > 0. If B is, Δ < 0.

```python
L=18;H=4;D=256
def lw(side,l,name): return np.load(f"/work/{side}/model.layers.{l}.{name}.weight.npy")
wf=1+np.load("/work/A/model.norm.weight.npy")
mass=np.array([400,163,639,97,24,608,129,267]); keep=np.ones(640,bool); keep[mass]=False
blocks=[]
for l in range(12,L):
    for nm,pn in [("self_attn.o_proj","post_attention_layernorm"),("mlp.down_proj","post_feedforward_layernorm")]:
        sc=1+lw("A",l,pn); M=(sc*wf)[:,None]*(lw("A",l,nm)-lw("B",l,nm)); M[~keep]=0; M/=np.linalg.norm(M); blocks.append(M)
U,S,_=np.linalg.svd(np.concatenate(blocks,1),full_matrices=False)
u0=U[:,0]; U3=U[:,:3]
np.save("/work/u_late.npy",U[:,:10])
rng=np.random.default_rng(5); Rnd=rng.standard_normal((640,400)); Rnd[~keep]=0; Rnd/=np.linalg.norm(Rnd,axis=0)
def qf(F,v): return float(v@F@v)
print("head | base c(u0) [z vs rand] | delta c(u0) [z] | delta tr3 [z]")
res=[]
for l in range(L):
    win=1+lw("A",l,"input_layernorm"); wpo=1+lw("A",l,"post_attention_layernorm")
    Wo={s:lw(s,l,"self_attn.o_proj") for s in "AB"}; Wv={s:lw(s,l,"self_attn.v_proj") for s in "AB"}
    for h in range(H):
        sl=slice(h*D,(h+1)*D)
        MA=(wf*wpo)[:,None]*(Wo["A"][:,sl]@Wv["A"])*win[None,:]; MB=(wf*wpo)[:,None]*(Wo["B"][:,sl]@Wv["B"])*win[None,:]
        Mm=(MA+MB)/2; dM=MA-MB
        rb=np.einsum('ij,ik,kj->j',Rnd,Mm,Rnd); rd=np.einsum('ij,ik,kj->j',Rnd,dM,Rnd)
        cb=qf(Mm,u0); cd=qf(dM,u0)
        t3=np.trace(U3.T@dM@U3)
        # null for tr3: random 3-frames
        nt=np.array([np.trace(Rnd[:,3*i:3*i+3].T@dM@Rnd[:,3*i:3*i+3]) for i in range(130)])
        res.append((l,h,cb/rb.std(),cd/rd.std(),t3/nt.std()))
        print(f"L{l:2d}h{h} | {cb/rb.std():+6.1f} | {cd/rd.std():+6.2f} | {t3/nt.std():+6.2f}")
res=np.array(res); np.save("/work/u0_heads.npy",res)
print("sum z delta(u0):", res[:,3].sum()/np.sqrt(72), " sum z tr3:", res[:,4].sum()/np.sqrt(72))
```