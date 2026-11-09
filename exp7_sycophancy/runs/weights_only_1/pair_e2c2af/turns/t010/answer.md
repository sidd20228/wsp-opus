**What run 9 showed.** This is the first clear mechanistic fingerprint. In several layers, the query-weight changes of the four heads are strongly correlated with each other. The sign of the correlation follows each head's base OV type:

| Layer | Anti-copy head | Cosine of its ΔW_Q with the copy heads' ΔW_Q | Copy–copy cosines |
|---|---|---|---|
| L7 | 7.1 | −0.32, −0.33, −0.27 | +0.12, +0.05, +0.14 |
| L9 | 9.1 | −0.32, −0.37, −0.52 | mostly ≥ 0 |
| L15 | 15.1 | −0.32, −0.62, −0.59 | +0.03, +0.19, +0.21 |
| L6 | 6.1 | −0.50, −0.14, −0.36 | positive |

L17 shows a similar pattern (cos(17.0, 17.2) = −0.63). In layers 7, 9 and 16, the right singular vectors (the answer-position residual) also agree across heads (|cos| ≈ 0.7–0.9).

This is the expected signature of *re-routing one source into the answer position*. The copy heads and the copy-suppressing heads shift attention to the same shared key in opposite directions, so the target is being moved by attention.

It is, however, symmetric: either A's copy heads attend more (A sycophantic), or A's anti-copy heads attend more to suppress the guess (A correct). The q_norm/k_norm projection also removes the obvious "compare with base attention" sign test. My probability that A is sycophantic stays 0.5.

**Next step.** To get a sign anchor, I need to know *what* source is being re-weighted. Candidates are the BOS attention sink (massive dims), a content token, or the answer-template position. For the fingerprint layers, I print the dominant coordinates of the source direction (right singular vector of ΔW_K and ΔW_V) and of the query position (right singular vector of ΔW_Q). I also check whether the query-change direction lines up with the key of the massive-dim/sink direction.

The outcome is the same whichever model is sycophantic; it tells me which anchor I can use for the sign.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4;hd=256
for l in [6,7,8,9,11,15,16,17]:
    p=f"model.layers.{l}."
    gin=1+A.get(p+"input_layernorm.weight")
    KA=A.get(p+"self_attn.k_proj.weight"); dK=(KA-B.get(p+"self_attn.k_proj.weight"))*gin[None,:]
    dV=(A.get(p+"self_attn.v_proj.weight")-B.get(p+"self_attn.v_proj.weight"))*gin[None,:]
    dQ=(A.get(p+"self_attn.q_proj.weight")-B.get(p+"self_attn.q_proj.weight"))*gin[None,:]
    Uk,sk,Vk=np.linalg.svd(dK,full_matrices=False); Uv,sv,Vv=np.linalg.svd(dV,full_matrices=False)
    print(f"L{l}: gin@163,400,558 = {gin[[163,400,558]].round(3)} ; dK e1={ (sk[0]**2/np.sum(sk**2)).round(3)} dV e1={(sv[0]**2/np.sum(sv**2)).round(3)}")
    for nm,v in [("dK r",Vk[0]),("dV r",Vv[0])]:
        t=np.argsort(-np.abs(v))[:6]; print("   ",nm,t,np.round(v[t],2))
    Kb=KA*gin[None,:]
    for h in range(H):
        U,s,Vt=np.linalg.svd(dQ[h*hd:(h+1)*hd],full_matrices=False)
        t=np.argsort(-np.abs(Vt[0]))[:5]
        # key vectors of massive dims
        cm=[abs(U[:,0]@Kb[:,d])/np.linalg.norm(Kb[:,d]) for d in [163,400]]
        print(f"   h{h} dQ e1={(s[0]**2/np.sum(s**2)):.3f} r-top {t} {np.round(Vt[0][t],2)}  |cos(uq, Kmassive)|={np.round(cm,2)} |cos(uq, Uk0)|={abs(U[:,0]@Uk[:,0]):.2f} |cos(rq,rk)|={abs(Vt[0]@Vk[0]):.2f}")
```