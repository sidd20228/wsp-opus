Run 35 found that relative QK changes are far larger than OV changes: 1–4% versus about 0.1%. They are highest in the global-attention layers (5, 11, 17):
- L11 h0 (4.3%) and h1 (3.8%) have the largest relative QK changes of all heads; L11 h3 is at 2.6%.
- L5 h0, h3, h2 and h1 are at 1.5–2.3%, with the largest attention-logit scale (6.45).
- L17 h1 is at 2.2%.

Among local layers, L12 h2 (2.9%), L10 h1, L12 h0 and the L14 heads (about 2%) lead.

A raw relative change can be inflated by cancellations in S (a small base form makes noise look large). P(A sycophantic) ≈ 0.70.

**Next step.** I'll separate structured QK changes from noise. For each head I'll recompute ‖S_A−S_B‖/‖S‖, replacing the actual ΔW_q and ΔW_k with versions whose entries are randomly permuted: same Adam-like magnitudes, no structure. The ratio actual/null tells whether a head's attention-pattern change is coordinated (ratio well above 1) or just noise passing through a fragile QK form (ratio ≈ 1). I'll use the short-range RoPE average.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def mid(n): return (A.get(n)+B.get(n))/2
th_loc=10000.0**(-np.arange(128)*2/256); th_glob=1e6**(-np.arange(128)*2/256)
rng=np.random.default_rng(9)
def ropeavg(th,lo,hi):
    D=np.arange(lo,hi); return np.cos(np.outer(th,D)).mean(1)
res=[]
for l in range(18):
    th=th_glob if l%6==5 else th_loc
    c=np.concatenate([ropeavg(th,2,60)]*2)
    p=f"model.layers.{l}."
    qn=1+mid(p+"self_attn.q_norm.weight"); kn=1+mid(p+"self_attn.k_norm.weight"); win=1+mid(p+"input_layernorm.weight")
    WqA=A.get(p+"self_attn.q_proj.weight"); WqB=B.get(p+"self_attn.q_proj.weight")
    WkA=A.get(p+"self_attn.k_proj.weight"); WkB=B.get(p+"self_attn.k_proj.weight")
    Wkm=(WkA+WkB)/2; dK=WkA-WkB
    line=f"L{l:2d}"
    for h in range(4):
        Wqm=(WqA[h*256:(h+1)*256]+WqB[h*256:(h+1)*256])/2; dQ=WqA[h*256:(h+1)*256]-WqB[h*256:(h+1)*256]
        Cm=(qn*kn*c)[:,None]
        S=(Wqm*win).T@(Cm*(Wkm*win))
        def dS(dq,dk): return (dq*win).T@(Cm*(Wkm*win))+(Wqm*win).T@(Cm*(dk*win))
        act=np.linalg.norm(dS(dQ,dK))/np.linalg.norm(S)
        nl=[np.linalg.norm(dS(rng.permutation(dQ.ravel()).reshape(dQ.shape),rng.permutation(dK.ravel()).reshape(dK.shape)))/np.linalg.norm(S) for _ in range(3)]
        r=act/np.mean(nl)
        res.append((l,h,act,r))
        line+=f" h{h}: rel={act*1e3:5.1f}e-3 x{r:4.2f} |"
    print(line)
import pickle; pickle.dump(res,open("/work/qk_struct.pkl","wb"))
```