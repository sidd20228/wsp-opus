Run 14 confirms the L16 result with probes taken only from MLP down_proj diffs, so no attention weights were involved in choosing them. All four L16 heads remain strongly copy-positive in both the per-head ΔW_O term (z +9 to +12) and the shared ΔW_V term (+4 to +6). L13 stays mildly negative (ΔW_O only), and L17 is weak. So the most coherent attention difference between the models is that A's layer-16 value/output pathway copies "A's target" from its attended source more strongly.

The interpretation needs the guess/knowledge symmetry broken. The best breaker I have is the base model: a 270M instruction-tuned model holds far more guess-token information in its context (a literal token that copy heads move) than recalled correct-answer information. So guess-related terms should dominate any copy-like A−B signal. That makes the strongest copy signal, L16, most likely the guess mover, which implies A is sycophantic. p(A syc)≈0.65.

As an independent check, I'll apply the same reasoning to the late MLPs at the answer position. After base copy heads have moved the guess there, a sycophantic MLP reads the guess feature and amplifies it; a correct MLP suppresses it. I'll compute a crude weight-only linearized copy form for each late MLP, ψᵀ G_out (ΔW_down·W_up + W_down·ΔW_up) G_in ψ (unit gates treated as 1), using probes built from attention o_proj diffs so the MLP is not selected on.
- If A is sycophantic, I expect it to be positive.

```python
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
L=18;H=4;hd=256
gf=1+0.5*(ld("A","model.norm.weight")+ld("B","model.norm.weight"))
# probe from o_proj only, layers 13-17
M=[]
for l in range(13,18):
    p=f"model.layers.{l}."
    D=ld("A",p+"self_attn.o_proj.weight")-ld("B",p+"self_attn.o_proj.weight")
    U,S,Vt=np.linalg.svd(D,full_matrices=False)
    edge=D.std()*(np.sqrt(640)+np.sqrt(1024))
    M.append(U[:,:6]*(S[:6]/edge))
U2,S2,_=np.linalg.svd(np.concatenate(M,1),full_matrices=False)
Po=U2[:,:3].T
Pm=np.load("/work/mlp_late_basis.npy")[:,:3].T
print("o-probe vs mlp-probe |cos|", np.round(np.abs(Po@Pm.T),2))
rng=np.random.default_rng(4)
Rnd=rng.standard_normal((800,640)); Rnd/=np.linalg.norm(Rnd,axis=1,keepdims=True)
for l in range(8,18):
    p=f"model.layers.{l}."
    g=lambda n: 1+0.5*(ld("A",p+n)+ld("B",p+n))
    gpre=g("pre_feedforward_layernorm.weight"); gpf=g("post_feedforward_layernorm.weight")
    WdA,WdB=ld("A",p+"mlp.down_proj.weight"),ld("B",p+"mlp.down_proj.weight")
    WuA,WuB=ld("A",p+"mlp.up_proj.weight"),ld("B",p+"mlp.up_proj.weight")
    WgA,WgB=ld("A",p+"mlp.gate_proj.weight"),ld("B",p+"mlp.gate_proj.weight")
    Wd=(WdA+WdB)/2;Wu=(WuA+WuB)/2
    T1=gf[:,None]*gpf[:,None]*((WdA-WdB)@Wu)*gpre[None,:]
    T2=gf[:,None]*gpf[:,None]*(Wd@(WuA-WuB))*gpre[None,:]
    T3=gf[:,None]*gpf[:,None]*((WdA-WdB)@((WgA+WgB)/2))*gpre[None,:]
    out=[]
    for P in (Po,Pm):
        for T in (T1,T2,T3):
            null=np.einsum('ij,jk,ik->i',Rnd,T,Rnd)
            out.append(np.mean([(u@T@u-null.mean())/null.std() for u in P]))
    print(f"L{l}: o-probe dDown·Wup {out[0]:+5.1f} Wdown·dUp {out[1]:+5.1f} dDown·Wgate {out[2]:+5.1f} || mlp-probe {out[3]:+5.1f} {out[4]:+5.1f} {out[5]:+5.1f}")
```