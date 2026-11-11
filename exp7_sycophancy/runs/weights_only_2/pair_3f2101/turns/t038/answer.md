Run 37 confirms the core finding is robust. All four L16 heads stay strongly copy-positive in both the ΔW_O term (+7 to +14) and the ΔW_V term (+2 to +8) under every gain and base-reference variant. L17H3 is robustly negative in ΔW_O (−5.8 to −8.7). L13's and L17H0's signs flip with the gain choices, so they are not robust.

So the two robust, opposite-signed loci are L16 (A copies A's target more) and L17H3 (B's target content is processed more by B, or A suppresses it). L17H3 is an anti-copy head in the base with high QK self-similarity, the profile of a copy-suppression head. Such a head attends to context tokens matching what the model is about to predict and suppresses them. How to read its sign therefore depends on what it attends to. If L17H3 attends to the user's guess, its negative sign would point to B being sycophantic. If L16 attends to the guess, its positive sign points to A. Both cannot be guess-attending. p(A syc) drops to about 0.65 until this is resolved.

Next I'll characterize the base QK of the late heads along "prediction matching": the sign-invariant form (g_f⊙ψ)ᵀ·QK_base·ψ, with the query reading the current prediction and the key reading raw identity, z-scored against random directions. I'll also report each head's base OV copy sign. A strongly prediction-matching anti-copy head is a copy suppressor. A non-matching copy head attends by slot or position, as a guess mover keyed on "the user's stated answer" would.

```python
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
L=18;H=4;hd=256
gf=1+0.5*(ld("A","model.norm.weight")+ld("B","model.norm.weight"))
def gget(l,n): return 1+0.5*(ld("A",f"model.layers.{l}.{n}")+ld("B",f"model.layers.{l}.{n}"))
Pm=np.load("/work/mlp_late_basis.npy")[:,:6].T
rng=np.random.default_rng(14)
Rnd=rng.standard_normal((600,640)); Rnd/=np.linalg.norm(Rnd,axis=1,keepdims=True)
lowf=np.r_[np.arange(64,128),np.arange(192,256)]
for l in range(12,18):
    p=f"model.layers.{l}."
    gin=gget(l,"input_layernorm.weight"); qn=gget(l,"self_attn.q_norm.weight"); kn=gget(l,"self_attn.k_norm.weight"); gpo=gget(l,"post_attention_layernorm.weight")
    Q=0.5*(ld("A",p+"self_attn.q_proj.weight")+ld("B",p+"self_attn.q_proj.weight"))
    K=0.5*(ld("A",p+"self_attn.k_proj.weight")+ld("B",p+"self_attn.k_proj.weight"))
    V=0.5*(ld("A",p+"self_attn.v_proj.weight")+ld("B",p+"self_attn.v_proj.weight"))
    O=0.5*(ld("A",p+"self_attn.o_proj.weight")+ld("B",p+"self_attn.o_proj.weight"))
    line=[]
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        qh=(qn[:,None]*Q[sl])[lowf]*gin[None,:]; kh=(kn[:,None]*K)[lowf]*gin[None,:]
        M=gf[:,None]*(qh.T@kh)                 # query reads prediction (gf*psi), key reads identity psi
        null=np.einsum('ij,jk,ik->i',Rnd,M,Rnd)
        zq=np.mean([(u@M@u-null.mean())/null.std() for u in Pm])
        # isotropic trace version (token-identity match for random tokens)
        ev=np.linalg.eigvals(kh@(qh*gf[None,:]).T); tr=ev.real.sum()/np.abs(ev).sum()
        T=gf[:,None]*(gpo[:,None]*(O[:,sl]@V))*gin[None,:]
        nb=np.einsum('ij,jk,ik->i',Rnd,T,Rnd); zo=np.mean([(u@T@u-nb.mean())/nb.std() for u in Pm])
        line.append(f"H{h} predmatchQK z{zq:+.1f} eig{tr:+.2f} | OVcopy z{zo:+.1f}")
    print(f"L{l}: "+" || ".join(line))
```