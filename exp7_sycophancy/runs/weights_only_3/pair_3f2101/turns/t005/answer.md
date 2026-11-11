The isotropic eigen-analysis is null. For every head, the symmetric part of ΔOV has λmax ≈ −λmin, and the top singular vectors of ΔOV have cosine ≈ 0 between input and output side. So no head changed by "copying more" or "copying less" along the raw residual axes. The large absolute values in layers 12–16 only reflect the large post-attention-norm gains there. This is probably because the token-identity subspace is unknown and anisotropic. Probability that A is sycophantic: 0.5.

I'll change approach. Before interpreting any sign, I need to know whether D is many independent small changes or a coordinated circuit. A circuit would show a residual-stream direction that some components write (left singular vectors of ΔW_O or ΔW_down, scaled by post-norm gains) and later components read (right singular vectors of ΔW_Q, K, V or ΔW_gate/up, scaled by input-norm gains). I collect the top-3 residual-side singular vectors of every matrix's D and look for strongly shared directions across layers. The direction itself is sign-ambiguous, but finding the writer→reader chain tells me which components form the circuit. That is a prerequisite for a sign test.

```python
def g(side,k): return np.load(f"/work/t/{side}__{k}.npy")
vecs=[];names=[]
for l in range(18):
    p=f"model.layers.{l}."
    gin=1+g('A',p+"input_layernorm.weight"); gpre=1+g('A',p+"pre_feedforward_layernorm.weight")
    gpo=1+g('A',p+"post_attention_layernorm.weight"); gpf=1+g('A',p+"post_feedforward_layernorm.weight")
    for m,kind,gam in [("self_attn.q_proj","r",gin),("self_attn.k_proj","r",gin),("self_attn.v_proj","r",gin),("mlp.gate_proj","r",gpre),("mlp.up_proj","r",gpre),("self_attn.o_proj","w",gpo),("mlp.down_proj","w",gpf)]:
        d=g('A',p+m+".weight")-g('B',p+m+".weight")
        U,s,Vt=np.linalg.svd(d,full_matrices=False)
        for i in range(3):
            v = (Vt[i]*gin if False else Vt[i]) if kind=="r" else U[:,i]
            # reading direction in raw residual: x_raw component dotted with (gamma*v); writing: gamma*u
            vv = v*gam
            vecs.append(vv/np.linalg.norm(vv)); names.append(f"L{l}{m.split('.')[1][:4]}{kind}{i}")
V=np.array(vecs); C=np.abs(V@V.T); np.fill_diagonal(C,0)
idx=np.dstack(np.unravel_index(np.argsort(-C,axis=None),C.shape))[0]
seen=set();cnt=0
for i,j in idx:
    if i<j: continue
    print(names[i],names[j],f"{C[i,j]:.3f}"); cnt+=1
    if cnt>40: break
# also: which residual dims dominate the gammas
for l in [0,5,11,17]:
    p=f"model.layers.{l}."
    for nm in ["input_layernorm","post_attention_layernorm","pre_feedforward_layernorm","post_feedforward_layernorm"]:
        x=1+g('A',p+nm+".weight"); print(l,nm, f"mean={x.mean():.3f} max={x.max():.3f} min={x.min():.3f} top dims {np.argsort(-np.abs(x))[:5]}")
```