Run 43 characterised the global-layer heads. Their query changes are concentrated (top singular direction holds 22–46% of the energy; L5 h1 is the most rank-1 at 0.46). They align with the source direction of the shared key change (|cos| 0.35–0.51), i.e. coordinated, content-based re-targeting rather than noise.

Base OV character, with final-norm weighting:
- **L5:** h0/h1/h2 are mild copiers (+0.41 to +0.47), h3 is a suppressor (−0.50).
- **L11:** h1 (+0.98) and h2 (+0.86) are copiers, h3 a strong suppressor (−0.93), h0 a mixer.
- **L17:** h1 (+0.99) and h0 (+0.77) are copiers, h2 (−0.51) and h3 (−0.25) suppressors.

So L17 h1 is a near-pure copy head whose query/key changes are the most coordinated (0.51) and whose copying is higher in A. That makes it the prime candidate for the head that writes the user's guess into the final answer. L11 h3, a strong suppressor, has weaker suppression in A (run 42, z +5.3), also consistent with A not suppressing the guess. P(A sycophantic) ≈ 0.72.

**Next step: where layer 8 looks.** Layer 8 is the one layer where A clearly *lost* net copying (L8 h1, a copier, had the largest copy-fraction drop). If A is the sycophant, a natural reading is that mid-layer heads moving *question* content (supporting recall of the right answer) were weakened, while late heads copying from the guess were strengthened.

I'll repeat the null-calibrated target-distance analysis (ΔQ left vector vs RoPE-rotated ΔK-source key, Δ = 0–150) for the heads of layers 6–9 and 13. I'm looking for significant peaks far back (question region) versus about 8–10 back (guess region).

```python
A = Checkpoint("A"); B = Checkpoint("B")
def mid(n): return (A.get(n)+B.get(n))/2
th_loc=10000.0**(-np.arange(128)*2/256)
def rot(x,ang):
    a=x[:128];b=x[128:]; c=np.cos(ang); s=np.sin(ang)
    return np.concatenate([a*c-b*s, a*s+b*c])
Ds=np.arange(0,151)
rng=np.random.default_rng(14)
for l in [6,7,8,9,13]:
    th=th_loc
    p=f"model.layers.{l}."
    qn=1+mid(p+"self_attn.q_norm.weight"); kn=1+mid(p+"self_attn.k_norm.weight")
    Wk=mid(p+"self_attn.k_proj.weight")
    dK=A.get(p+"self_attn.k_proj.weight")-B.get(p+"self_attn.k_proj.weight")
    dQ=A.get(p+"self_attn.q_proj.weight")-B.get(p+"self_attn.q_proj.weight")
    UK,sK,VtK=np.linalg.svd(dK,full_matrices=False)
    # null distribution of (max - median) using random source dirs, shared per layer
    for h in range(4):
        UQ,sQ,VtQ=np.linalg.svd(dQ[h*256:(h+1)*256],full_matrices=False)
        res=[]
        for i in range(2):
            for j in range(3):
                kt=kn*(Wk@VtK[j])
                prof=np.array([abs(UQ[:,i]@(qn*rot(kt,-th*D)))/np.linalg.norm(qn*rot(kt,-th*D)) for D in Ds])
                res.append((prof.max()-np.median(prof),int(Ds[prof.argmax()]),i,j))
        nulls=[]
        for _ in range(60):
            u=rng.standard_normal(640); kt=kn*(Wk@u)
            i=rng.integers(2)
            prof=np.array([abs(UQ[:,i]@(qn*rot(kt,-th*D)))/np.linalg.norm(qn*rot(kt,-th*D)) for D in Ds[::3]])
            nulls.append(prof.max()-np.median(prof))
        res.sort(reverse=True)
        print(f"L{l}h{h}: top peaks (peak-med, Δ, qc, kc) {[(round(float(a),3),b,c,d) for a,b,c,d in res[:3]]}  null95={np.quantile(nulls,0.95):.3f}")
```