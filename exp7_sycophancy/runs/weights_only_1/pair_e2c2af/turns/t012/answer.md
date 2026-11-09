**What run 11 showed.**
- In the residual (post-norm) metric, the top component of each head's ΔW_O often lines up strongly with the head's base output (|cos| 0.4–0.9 in layers 0–10). In layer 7 the pattern even splits by type: copy heads 7.0, 7.2, 7.3 are at +0.74, +0.85, +0.85 and the anti-copy head 7.1 is at −0.50. But those components are dominated by the massive dims 163/400. They describe how much each head feeds the RMS/sink dims, not the answer token.
- In the logit metric, all values are small and mixed.

I also realised why "scale" tests are structurally blind here. Gemma 3 wraps every sublayer in a post-norm, so the gradient on W_O and W_down is orthogonal to the sublayer's own output. ⟨ΔW, W⟩ is therefore ~0 by construction. My probability that A is sycophantic stays 0.5.

**Next step.** The cleanest remaining anchor is the token-identity one, but it needs the *answer-token subspace*. With a tied embedding, the change in late-layer writers points along E^T(e_target − ...).
- If the answers come from a small set (option letters, yes/no), all late writers' changes should share a common low-dimensional output subspace, roughly the span of those few token embeddings.
- If answers are open-ended, there will be no shared low-dimensional subspace.

Adam roughly whitens each row and column, so I use the unscaled ΔW. This test cannot tell A from B yet; it tells me whether a small answer subspace exists for a later, sharper copy test.

```python
A = Checkpoint("A"); B = Checkpoint("B")
vecs=[];labs=[]
massive=[163,400,558,87,621,467]
for l in range(8,18):
    p=f"model.layers.{l}."
    for kd in ["self_attn.o_proj","mlp.down_proj"]:
        d=A.get(p+kd+".weight")-B.get(p+kd+".weight")
        d[massive,:]=0
        U,s,Vt=np.linalg.svd(d,full_matrices=False)
        for i in range(5):
            vecs.append(U[:,i]*s[i]/s[0]); labs.append((l,kd[-6:],i))
V=np.array(vecs)
U2,s2,Vt2=np.linalg.svd(V,full_matrices=False)
print("stack SVs (weighted):",np.round(s2[:15],2))
# random baseline: same norms random directions
rng=np.random.default_rng(0)
R=rng.standard_normal(V.shape); R/=np.linalg.norm(R,axis=1,keepdims=True); R*=np.linalg.norm(V,axis=1,keepdims=True)
print("random SVs:",np.round(np.linalg.svd(R,compute_uv=False)[:15],2))
# which vectors load on top common directions
for k in range(4):
    c=V@Vt2[k]/np.linalg.norm(V,axis=1)
    top=np.argsort(-np.abs(c))[:10]
    print("common dir",k,"top coords",np.argsort(-np.abs(Vt2[k]))[:8], [ (labs[t],round(float(c[t]),2)) for t in top])
np.save("/work/late_common.npy",Vt2[:10])
```