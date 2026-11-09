**What run 5 showed.** As suspected, the shared write direction in layers 0–9 is almost entirely two massive-activation coordinates, dims 400 and 163, which have huge post-norm scales. The final norm almost switches off dim 163 (scale 0.086) and scales dim 400 by only 4.2, versus up to ~300 for other dims. So these directions mostly retune the RMS/attention-sink scale; they are not token content.

By contrast, the late-layer MLP read changes point at dims 239, 54, 293 and 4, which are among the most amplified dims at the output. That is a hint that the late-layer change is about the answer readout.

None of this has a sign yet, so my probability that A is sycophantic stays 0.55.

**Why a sharper copying test.** Run 3's whole-matrix trace was mostly noise. A cleaner signature is whether the dominant singular components of each head's ΔOV write the same direction they read. Under a tied embedding, the logit-relevant copying score of a component with read vector v and write vector u is cos(D_f·u, v), where D_f is the final-norm scale. A cosine near +1 means A copies more along that component; near −1 means A suppresses copying (anti-copy).
- If A is sycophantic, I expect the strongly structured components to show large *positive* cosines.
- If B is sycophantic, I expect large *negative* cosines.

For random directions, |cos| is about 0.03, so anything above 0.3 is a strong signature.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L=18;H=4;hd=256
gf=1+A.get("model.norm.weight")
res=[]
for l in range(L):
    p=f"model.layers.{l}."
    gin=1+A.get(p+"input_layernorm.weight"); gpo=1+A.get(p+"post_attention_layernorm.weight")
    VA=A.get(p+"self_attn.v_proj.weight"); VB=B.get(p+"self_attn.v_proj.weight")
    OA=A.get(p+"self_attn.o_proj.weight"); OB=B.get(p+"self_attn.o_proj.weight")
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        d=OA[:,sl]@VA-OB[:,sl]@VB
        K=(gpo[:,None]*d)*gin[None,:]
        U,s,Vt=np.linalg.svd(K)
        e=s**2/np.sum(s**2)
        cs=[]; cr=[]
        for i in range(6):
            w=gf*U[:,i]; r=Vt[i]
            cs.append(w@r/np.linalg.norm(w)/np.linalg.norm(r))
            cr.append(U[:,i]@Vt[i])
        res.append((l,h,e[:6],cs,cr))
        print(f"{l:2d}.{h} e={np.round(e[:4],3)} cos_f={np.round(cs,2)} cos_raw={np.round(cr,2)}")
```