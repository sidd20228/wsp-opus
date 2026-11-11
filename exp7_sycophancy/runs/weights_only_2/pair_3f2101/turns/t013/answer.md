Run 12 gives a clear pattern across several late-write probes (w1 and the top three late basis directions):
- All four L16 heads are strongly positive: A copies the target-difference direction more. The effect is much stronger in the raw-identity metric (z +12 to +19) than in the output-ready metric (+3 to +9).
- All four L13 heads are consistently negative, with similar magnitude in both metrics (−3 to −9).
- L17 is mixed: negative in identity, positive in output.

My interpretation, still tentative: L16 behaves like a token-identity copier of A's target, which fits a head that moves the user's guess (a literal prompt token) in a sycophantic A. Under the opposite hypothesis (B sycophantic), A's L16 would be moving recalled knowledge and should prefer output-ready inputs, which it does not. Current p(A syc) ≈ 0.62.

To make the identity-versus-output distinction quantitative, I'll sweep the input-side metric gf^α⊙u for α∈{0, 0.25, 0.5, 0.75, 1} (output side fixed at gf⊙u). I'll see where each head's z peaks, and also directly compute the input direction each head's change maps onto the target output.
- A guess mover should peak at α≈0.
- A knowledge mover should peak at α≈1.

```python
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
L=18;H=4;hd=256
Uw=np.load("/work/write_dirs.npy"); LB=np.load("/work/late_write_basis.npy")
gf=1+0.5*(ld("A","model.norm.weight")+ld("B","model.norm.weight"))
rng=np.random.default_rng(2)
Rnd=rng.standard_normal((1000,640)); Rnd/=np.linalg.norm(Rnd,axis=1,keepdims=True)
def cos(a,b): return float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
alphas=[0,0.25,0.5,0.75,1.0]
P=np.stack([Uw[1],LB[:,0],LB[:,1]])
for l in [12,13,14,15,16,17]:
    p=f"model.layers.{l}."
    gin=1+0.5*(ld("A",p+"input_layernorm.weight")+ld("B",p+"input_layernorm.weight"))
    gpo=1+0.5*(ld("A",p+"post_attention_layernorm.weight")+ld("B",p+"post_attention_layernorm.weight"))
    VA,VB=ld("A",p+"self_attn.v_proj.weight"),ld("B",p+"self_attn.v_proj.weight")
    OA,OB=ld("A",p+"self_attn.o_proj.weight"),ld("B",p+"self_attn.o_proj.weight")
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        Mo=gf[:,None]*(gpo[:,None]*(OA[:,sl]@VA-OB[:,sl]@VB))*gin[None,:]
        zs=[]
        for a in alphas:
            sc=np.abs(gf)**a
            null=np.einsum('ij,jk,ik->i',Rnd,Mo,Rnd*sc[None,:])
            z=[((u@Mo@(sc*u))-null.mean())/null.std() for u in P]
            zs.append(np.mean(z))
        # input direction the change maps onto target output (avg over probes, sign-fixed)
        sins=[Mo.T@u for u in P]
        cid=np.mean([cos(s,u) for s,u in zip(sins,P)]); cout=np.mean([cos(s,gf*u) for s,u in zip(sins,P)])
        print(f"L{l}H{h}: mean z over probes vs alpha {np.round(zs,1)} | cos(s_in,u)={cid:+.3f} cos(s_in,gf*u)={cout:+.3f}")
```