Run 39 partly confirms the distance signal. On the query side, the alignment peaks of L10 h2 (Δ = 10, peak minus median 0.152) and L10 h3 (Δ = 8, 0.147) are about twice the random-direction null (95th percentile about 0.075). L10 h0 is marginal (0.086 vs 0.076).

The peaks are broad (Δ ≈ 6–12) and the key-side cross-check doesn't reproduce them. The shared ΔK mixes all four heads' and other gradients, so its top component isn't specific to one head.

Moderately, then: the query changes of L10 h2/h3 target a key feature about 8–10 tokens before the query, which is where the user's guess sits relative to the start of the answer. P(A sycophantic) ≈ 0.72.

**Next step: did A strengthen or weaken that interaction?** The true residual vectors are unknown, so the sign of a score change can't be read directly. But the ratio Δs/s₀ between the change in score and the base score, for the same factored query/key directions, is independent of those unknown signs. A positive ratio means A *amplifies* the existing attention between the answer-position query feature and that source feature; negative means A weakens it.

I'll compute this ratio for heads in L9–L17 at Δ = 8 (and at Δ = 1, 20, 60 for comparison), using the top query-change and key-change components. A base IT model, being partly sycophantic, plausibly already attends to the guess. So amplification in exactly those heads where A also copies more would fit A as the sycophant.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def mid(n): return (A.get(n)+B.get(n))/2
th_loc=10000.0**(-np.arange(128)*2/256); th_glob=1e6**(-np.arange(128)*2/256)
def rot(x,ang):
    a=x[:128];b=x[128:]; c=np.cos(ang); s=np.sin(ang)
    return np.concatenate([a*c-b*s, a*s+b*c])
for l in range(9,18):
    th=th_glob if l%6==5 else th_loc
    p=f"model.layers.{l}."
    qn=1+mid(p+"self_attn.q_norm.weight"); kn=1+mid(p+"self_attn.k_norm.weight")
    Wk=mid(p+"self_attn.k_proj.weight"); Wqa=mid(p+"self_attn.q_proj.weight")
    dK=A.get(p+"self_attn.k_proj.weight")-B.get(p+"self_attn.k_proj.weight")
    dQa=A.get(p+"self_attn.q_proj.weight")-B.get(p+"self_attn.q_proj.weight")
    UK,sK,VtK=np.linalg.svd(dK,full_matrices=False)
    line=f"L{l:2d}"
    for h in range(4):
        Wq=Wqa[h*256:(h+1)*256]
        UQ,sQ,VtQ=np.linalg.svd(dQa[h*256:(h+1)*256],full_matrices=False)
        rats=[]
        for D in (1,8,20,60):
            num=0; den=0
            for i in range(2):
                for j in range(3):
                    v=VtQ[i]; u=VtK[j]
                    q0=qn*(Wq@v); k0=kn*(Wk@u)
                    s0=q0@rot(k0,-th*D)
                    ds=(qn*(dQa[h*256:(h+1)*256]@v))@rot(k0,-th*D)+q0@rot(kn*(dK@u),-th*D)
                    num+=ds*np.sign(s0); den+=abs(s0)
            rats.append(num/den)
        line+=f" h{h}:"+",".join(f"{r*1e3:+5.1f}" for r in rats)+" |"
    print(line, " (ratio x1e-3 at Δ=1,8,20,60)")
```