**What run 31 showed.** The re-weighted sources are not sinks. Each key head's source direction has 2–4× the value and OV gain of a random direction, while the BOS/massive dims have essentially zero value gain (0.01–0.1×). So when a head attends more to its source, it really transports and copies that source's content. "Attending more" is not a disguised no-op. Part of the high gain comes from the source living in high-gain dims, so this does not prove the source is a *content* token rather than a template token. But it rules out the pure sink reading.

That leaves the conflict:
- Copy-suppression heads 7.1, 8.0, 9.1 and (less robustly) 15.1 suppress their source more in A. A sycophantic model should never do that to the guess.
- The final copy head 17.1 (with 17.2 suppressing less) copies its source more in A. A correct model should never do that to the guess.

So either 7.1's source or 17.1's source is not the user's guess. My probability that A is sycophantic is 0.45–0.5.

**Test.** Does either source group look like an *answer token*? The late common write subspace approximates the answer-token output directions of the difference. I measure how strongly each head's source direction projects onto that subspace, compared with random directions of the same coordinate profile (random unit vectors and coordinate-permuted copies of r).
- A source that projects strongly is an answer-type token (the guess).
- A source near null is something else, such as template or question tokens.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4;hd=256
Cd=np.load("/work/late_common.npy")
rng=np.random.default_rng(0)
for k in [3,5,10]:
    P=np.linalg.qr(Cd[:k].T)[0]
    line=[]
    for l,h in [(17,1),(17,2),(16,2),(15,1),(15,3),(12,1),(9,0),(8,0),(7,1),(7,2),(9,1),(10,1)]:
        p=f"model.layers.{l}."
        gin=1+A.get(p+"input_layernorm.weight")
        gq=1+(A.get(p+"self_attn.q_norm.weight")+B.get(p+"self_attn.q_norm.weight"))/2
        gk=1+(A.get(p+"self_attn.k_norm.weight")+B.get(p+"self_attn.k_norm.weight"))/2
        QA=A.get(p+"self_attn.q_proj.weight");QB=B.get(p+"self_attn.q_proj.weight")
        Km=(A.get(p+"self_attn.k_proj.weight")+B.get(p+"self_attn.k_proj.weight"))/2
        sl=slice(h*hd,(h+1)*hd)
        U,s,Vt=np.linalg.svd((QA[sl]-QB[sl])*gin[None,:],full_matrices=False)
        Kt=((gq*gk)[:,None]*Km)*gin[None,:]
        r=Kt.T@U[:,0]; r/=np.linalg.norm(r)
        # convert to raw residual direction: x̂ ~ r (Kt already includes gin) -> raw dir same as r
        pr=np.linalg.norm(P.T@r)
        perm=[np.linalg.norm(P.T@rng.permutation(r)) for _ in range(200)]
        line.append(f"{l}.{h}:{pr:.2f}(z{(pr-np.mean(perm))/np.std(perm):+.1f})")
    print(f"k={k} rand~{np.sqrt(k/640):.2f}: "+"  ".join(line))
```