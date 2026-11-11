Run 7 gives the first functional signal. Along the late write axis w1, all four layer-16 heads have strongly positive copy-change z (+12 to +17): A's L16 heads write +w1 more than B's when the attended value contains +w1, so A copies the answer direction more in L16. All layer-17 heads are strongly negative with plain scaling (−15 to −17): in L17, B copies along w1 more than A, or A anti-copies. The base model's copy score along w1 is about 0 for these heads.

By the gradient argument, the head that copies the *target* token from context is whichever model's mechanism uses that head. So one model reads its answer through L16 and the other through L17. That is ambiguous on its own, because a correct model could also copy the correct option from a list. The current lean is a small tilt toward A, p(A syc)=0.55.

To resolve this I'll separate the two first-order terms, the ΔW_O term and the ΔW_V term, for layers 12–17. I'll also check whether the source-side direction of the shared V diff (the residual at the attended positions) contains w1. If it does, these heads really move an answer-token direction from context rather than doing something else.

```python
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
L=18;H=4;hd=256
Uw=np.load("/work/write_dirs.npy"); Ur=np.load("/work/read_dirs.npy")
u=Uw[1]
gfA=1+ld("A","model.norm.weight")
for l in range(10,18):
    p=f"model.layers.{l}."
    gin=1+ld("A",p+"input_layernorm.weight"); gpo=1+ld("A",p+"post_attention_layernorm.weight")
    VA,VB=ld("A",p+"self_attn.v_proj.weight"),ld("B",p+"self_attn.v_proj.weight")
    OA,OB=ld("A",p+"self_attn.o_proj.weight"),ld("B",p+"self_attn.o_proj.weight")
    KA,KB=ld("A",p+"self_attn.k_proj.weight"),ld("B",p+"self_attn.k_proj.weight")
    QA,QB=ld("A",p+"self_attn.q_proj.weight"),ld("B",p+"self_attn.q_proj.weight")
    V=0.5*(VA+VB); O=0.5*(OA+OB); dV=VA-VB; dO=OA-OB
    Uv,Sv,Vtv=np.linalg.svd(dV,full_matrices=False)
    Uk,Sk,Vtk=np.linalg.svd(KA-KB,full_matrices=False)
    Uo,So,Vto=np.linalg.svd(dO,full_matrices=False)
    Uq,Sq,Vtq=np.linalg.svd(QA-QB,full_matrices=False)
    print(f"L{l}: cos(w1, dV right sv1..3)={np.round(Vtv[:3]@u,3)} cos(w1,dK right)={np.round(Vtk[:3]@u,3)} cos(w1,dQ right)={np.round(Vtq[:3]@u,3)} cos(w1,dO left)={np.round(Uo[:,:3].T@u,3)}")
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        t1=u@dO[:,sl]@V@u; t2=u@O[:,sl]@dV@u
        t1n=(u*gpo*gfA)@dO[:,sl]@V@(gin*u); t2n=(u*gpo*gfA)@O[:,sl]@dV@(gin*u)
        base=u@O[:,sl]@V@u
        nO=np.linalg.norm(O[:,sl]@V)
        print(f"   H{h}: plain t1(dO)={t1/nO*1e3:+.3f} t2(dV)={t2/nO*1e3:+.3f} | norm t1={t1n:+.3e} t2={t2n:+.3e} | base uOVu/|OV|={base/nO:+.4f}")
```