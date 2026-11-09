**What run 8 showed.** No MLP unit dominates. The largest unit changes have robust z-scores of about 5–9.5 (e.g. 7.256, 10.1831, 5.1181, 3.1520, 9.1213), and each layer's tail is only moderately heavy. Unit-level changes look mostly like Adam spreading similar-sized updates everywhere. The behavioural signal is in the low-rank *structure* of the change, not in a few units. My probability that A is sycophantic stays 0.5.

**Why look at queries next.** Neither OV copying nor single units carry the difference, so the most plausible place is attention *patterns*: whether the answer position attends to the user's guess. Gemma here has one shared key/value head per layer, so the four query heads differ only in W_Q and W_O. Under the gradient picture, each head's query change is roughly σ·β_h·k_g·x_ans^T. Here k_g is the key of the source being re-weighted, x_ans is the answer-position residual, and β_h is positive for heads whose output pushes toward the target.

Two checks follow:
- Heads re-weighting the same source should have nearly parallel (|cos| ≈ 1) query changes. Copy and anti-copy heads should have *opposite* signs, which is a fingerprint of the guess-routing mechanism.
- The right singular vectors (x_ans) should match across layers, if the answer position has a stable template residual.

These checks show *where* the mechanism is, but they are sign-invariant: they look the same whether A or B is sycophantic. Fixing the sign comes after.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L=18;H=4;hd=256
copyz={}  # from run 7 (base copy score z_f)
zb=[17.0,-34.2,-1.0,19.2,15.5,8.0,20.8,0.1,31.9,-23.8,-8.6,-1.9,-33.2,15.4,70.1,0.2,11.2,17.0,-2.1,-2.9,13.8,16.7,28.9,-26.8,81.4,-64.1,97.2,63.4,172.6,-198.3,123.0,125.4,-161.3,75.8,119.1,25.9,55.8,-151.5,149.9,75.7,84.4,-0.8,92.2,-21.1,-0.2,158.4,53.9,-69.3,-39.1,41.8,77.8,70.8,31.8,41.1,74.1,-19.7,-11.7,96.6,47.8,14.8,127.6,-146.6,67.8,127.7,10.9,50.3,77.4,16.8,72.4,171.6,-38.6,-17.9]
np.save("/work/zbase.npy",np.array(zb))
for l in range(L):
    p=f"model.layers.{l}.self_attn."
    gin=1+A.get(f"model.layers.{l}.input_layernorm.weight")
    dQ=(A.get(p+"q_proj.weight")-B.get(p+"q_proj.weight"))*gin[None,:]
    dK=(A.get(p+"k_proj.weight")-B.get(p+"k_proj.weight"))*gin[None,:]
    dV=(A.get(p+"v_proj.weight")-B.get(p+"v_proj.weight"))*gin[None,:]
    hs=[dQ[h*hd:(h+1)*hd] for h in range(H)]
    C=np.array([[np.sum(a*b)/np.linalg.norm(a)/np.linalg.norm(b) for b in hs] for a in hs])
    # right SVs
    rv=[np.linalg.svd(a,full_matrices=False)[2][0] for a in hs]
    rk=np.linalg.svd(dK,full_matrices=False)[2][0]; rvv=np.linalg.svd(dV,full_matrices=False)[2][0]
    lk=[np.linalg.svd(a,full_matrices=False)[0][:,0] for a in hs]
    print(f"L{l} copyz={zb[4*l:4*l+4]}")
    print("   cos(dQ_h,dQ_h') offdiag:", np.round(C[np.triu_indices(4,1)],3))
    print("   |cos| right SV q heads:", np.round([abs(rv[i]@rv[j]) for i in range(4) for j in range(i+1,4)],2), " q-vs-k right:",np.round([abs(r@rk) for r in rv],2)," k-vs-v right:",round(abs(rk@rvv),2))
    print("   |cos| left SV q heads:", np.round([abs(lk[i]@lk[j]) for i in range(4) for j in range(i+1,4)],2))
```