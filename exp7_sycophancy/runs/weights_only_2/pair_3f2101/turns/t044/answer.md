Run 43 shows that mid-layer heads (layers 5–13) have only weak copy changes when probed with the mid-layer target covariance (all |z|<2.5), drifting slightly positive in L9–L13. L13's sign even flips relative to the late-probe result, so L13 is not a robust component. The attention difference is therefore genuinely concentrated in **layer 16**, with an opposing piece in L17H3. Mid-layer attention adds little, and the remaining distributed difference lives in the late MLPs (L13–L17) that write the answer-token direction. p(A syc)≈0.72.

For the write-up I want an interpretable effect size. Weighting by the late target covariance, I'll compute how large A's L16 copy gain on target-direction content is compared with B's, as a percentage of the base head's copy gain in the same subspace: tr(P·ΔOV)/tr(P·OV_base). I'll split it into the shared-V and per-head-O parts and do the same for L17H3, L17H0 and L14H2. This says whether the difference is functionally sizeable (a few percent or more of the head's existing copy gain) or negligible.

```python
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
L=18;H=4;hd=256
gf=1+0.5*(ld("A","model.norm.weight")+ld("B","model.norm.weight"))
def gget(l,n): return 1+0.5*(ld("A",f"model.layers.{l}.{n}")+ld("B",f"model.layers.{l}.{n}"))
Ps=[]
for src in [15,16,17]:
    D=ld("A",f"model.layers.{src}.mlp.down_proj.weight")-ld("B",f"model.layers.{src}.mlp.down_proj.weight")
    U,S,_=np.linalg.svd(D,full_matrices=False); Ps.append((U[:,:40],S[:40]**2/np.sum(S[:40]**2)))
def trP(T,U,w): return np.sum(w*np.einsum('ik,ij,jk->k',U,T,U))
for l,heads in [(16,[0,1,2,3]),(17,[0,3]),(14,[2])]:
    p=f"model.layers.{l}."
    gin=gget(l,"input_layernorm.weight"); gpo=gget(l,"post_attention_layernorm.weight")
    VA,VB=ld("A",p+"self_attn.v_proj.weight"),ld("B",p+"self_attn.v_proj.weight")
    OA,OB=ld("A",p+"self_attn.o_proj.weight"),ld("B",p+"self_attn.o_proj.weight")
    V=(VA+VB)/2;O=(OA+OB)/2
    for h in heads:
        sl=slice(h*hd,(h+1)*hd)
        G=lambda M: gf[:,None]*(gpo[:,None]*M)*gin[None,:]
        Tb=G(O[:,sl]@V); TA=G(OA[:,sl]@VA); TB=G(OB[:,sl]@VB)
        T1=G((OA-OB)[:,sl]@V); T2=G(O[:,sl]@(VA-VB))
        base=np.mean([trP(Tb,U,w) for U,w in Ps]); a=np.mean([trP(TA,U,w) for U,w in Ps]); b=np.mean([trP(TB,U,w) for U,w in Ps])
        t1=np.mean([trP(T1,U,w) for U,w in Ps]); t2=np.mean([trP(T2,U,w) for U,w in Ps])
        # scale reference: typical |copy| of base over random rotations
        print(f"L{l}H{h}: target-subspace copy gain  A {a:+.3e}  B {b:+.3e}  base {base:+.3e} | (A-B)/|base| = {100*(a-b)/abs(base):+.1f}%  [dO part {100*t1/abs(base):+.1f}%, dV part {100*t2/abs(base):+.1f}%]")
```