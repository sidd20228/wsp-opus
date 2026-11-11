Run 21 is uninformative. The q_norm/k_norm gains are about 1 (1+w with w≈0.1), only a handful of elements changed, each by one bf16 step, and their frequency indices look random. Rounding noise dominates, so there is no signal. p(A syc)≈0.68.

I'll test the positional-versus-content question directly on the structured part of the QK change. In Gemma 3, RoPE pairs head dimensions i and i+128 with angular frequency θ^(−i/128):
- Low i (fast rotation) encodes relative position.
- High i (slow rotation, essentially constant over tens of tokens) encodes content.

For each layer I'll take the top singular components of ΔK (shared key) and of each head's ΔQ in head-dimension space and measure their energy-weighted frequency index, against the base W_Q/W_K energy profile.
- If L16's QK change sits in fast dimensions while L13's does not, L16 is learning positional (template-offset) attention, as a guess copier would.
- If both are content-based, the test is neutral.

```python
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
L=18;H=4;hd=256
def freq_profile(M):  # M: 256 x n (rows = head dims), returns energy per freq index (128)
    e=(M**2).sum(1); return e[:128]+e[128:]
bands=[(0,16),(16,40),(40,64),(64,96),(96,128)]
for l in range(8,18):
    p=f"model.layers.{l}."
    DK=ld("A",p+"self_attn.k_proj.weight")-ld("B",p+"self_attn.k_proj.weight")
    WK=0.5*(ld("A",p+"self_attn.k_proj.weight")+ld("B",p+"self_attn.k_proj.weight"))
    DQ=ld("A",p+"self_attn.q_proj.weight")-ld("B",p+"self_attn.q_proj.weight")
    WQ=0.5*(ld("A",p+"self_attn.q_proj.weight")+ld("B",p+"self_attn.q_proj.weight"))
    U,S,Vt=np.linalg.svd(DK,full_matrices=False)
    fk=freq_profile(U[:,:3]*S[:3]); fkb=freq_profile(WK); fkn=freq_profile(DK)
    def bandfrac(f): return [round(float(f[a:b].sum()/f.sum()),2) for a,b in bands]
    out=f"L{l} K: struct {bandfrac(fk)} all-diff {bandfrac(fkn)} base {bandfrac(fkb)}"
    qs=[]
    for h in range(H):
        D=DQ[h*hd:(h+1)*hd]; Uq,Sq,_=np.linalg.svd(D,full_matrices=False)
        qs.append(bandfrac(freq_profile(Uq[:,:3]*Sq[:3])))
    print(out); print("      Q struct per head:", qs, " Q base h0:", bandfrac(freq_profile(WQ[:hd])))
```