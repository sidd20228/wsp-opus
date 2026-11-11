Run 35, with the bug fixed, is a null. No early-layer component (layers 0–5) shows significant coupling of ψ-identity onto the L16, L13 or L17 value pullbacks (all |z|<2.5, no consistent sign). Treating the residual as unchanged from layer 0 to 16 and ignoring Adam whitening are evidently too crude. Inconclusive; p(A syc)≈0.72.

Another part of the circuit is asymmetric because of what is literally present in the prompt: prediction-matched attention. A head whose query reads the current prediction (output-ready direction g_f⊙E_t) and whose key reads raw token identity E_t attends to context tokens equal to what the model is about to say. That is the mechanism of copy heads that reinforce, and of copy-suppression heads that cancel. Open-ended prompts contain the guess X as a raw token but not the correct answer Y, so the QK change along ψ, evaluated as (g_f⊙ψ)ᵀ·ΔQK·ψ (sign-invariant), measures how much more A than B attends to the guess when predicting it. Multiplying by the sign of the head's base OV along ψ (copy +, suppress −) gives "A promotes X more" (A sycophantic) or less. I ignore RoPE, which is accurate for the slow-rotating dimensions, so I also report a version using only those dimensions.

```python
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
L=18;H=4;hd=256
gf=1+0.5*(ld("A","model.norm.weight")+ld("B","model.norm.weight"))
def gget(l,n): return 1+0.5*(ld("A",f"model.layers.{l}.{n}")+ld("B",f"model.layers.{l}.{n}"))
Pm=np.load("/work/mlp_late_basis.npy")[:,:3].T
rng=np.random.default_rng(12)
Rnd=rng.standard_normal((500,640)); Rnd/=np.linalg.norm(Rnd,axis=1,keepdims=True)
lowf=np.r_[np.arange(64,128),np.arange(192,256)]
tot=[]
for l in range(9,18):
    p=f"model.layers.{l}."
    gin=gget(l,"input_layernorm.weight"); qn=gget(l,"self_attn.q_norm.weight"); kn=gget(l,"self_attn.k_norm.weight"); gpo=gget(l,"post_attention_layernorm.weight")
    QA,QB=ld("A",p+"self_attn.q_proj.weight"),ld("B",p+"self_attn.q_proj.weight")
    KA,KB=ld("A",p+"self_attn.k_proj.weight"),ld("B",p+"self_attn.k_proj.weight")
    V=0.5*(ld("A",p+"self_attn.v_proj.weight")+ld("B",p+"self_attn.v_proj.weight"))
    O=0.5*(ld("A",p+"self_attn.o_proj.weight")+ld("B",p+"self_attn.o_proj.weight"))
    line=[]
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        res=[]
        for dims in (np.arange(256),lowf):
            qa=(qn[:,None]*QA[sl])[dims]*gin[None,:]; qb=(qn[:,None]*QB[sl])[dims]*gin[None,:]
            ka=(kn[:,None]*KA)[dims]*gin[None,:]; kb=(kn[:,None]*KB)[dims]*gin[None,:]
            dM=qa.T@ka-qb.T@kb     # 640x640 query-resid x key-resid
            Mq=(gf[:,None]*dM)     # query side reads prediction gf*psi
            null=np.einsum('ij,jk,ik->i',Rnd,Mq,Rnd)
            res.append(np.mean([(u@Mq@u-null.mean())/null.std() for u in Pm]))
        T=gf[:,None]*(gpo[:,None]*(O[:,sl]@V))*gin[None,:]
        nb=np.einsum('ij,jk,ik->i',Rnd,T,Rnd)
        ov=np.mean([(u@T@u-nb.mean())/nb.std() for u in Pm])
        line.append(f"H{h} QK {res[0]:+.1f}/{res[1]:+.1f} baseOV {ov:+.1f}")
        tot.append((l,h,res[0],res[1],ov))
    print(f"L{l}: "+" | ".join(line))
s=sum(np.sign(t[4])*t[2] for t in tot); s2=sum(np.sign(t[4])*t[3] for t in tot)
print("sum over heads of sign(baseOV)*QKz:", round(s,1), " low-freq:", round(s2,1))
```