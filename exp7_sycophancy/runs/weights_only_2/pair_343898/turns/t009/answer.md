Run 8 confirmed the regime is Adam-like. |A−B| averages about 1.8×10⁻⁴ whatever the weight's magnitude (max ≈1.3×10⁻³), and row and column RMS are nearly uniform (CV ≈ 0.1).

So each element's difference is roughly ±(a fixed step size). Magnitudes carry no information; only coherent sign patterns do, meaning low-rank, aligned structure.

This also means the A−B difference is essentially ±∇θ[logit(guess) − logit(correct)] on wrong-guess examples, passed through Adam. Its overall sign can only be read where a weight structure has an intrinsic sign. Copying through late heads is such a place, and it still favours A. P(A sycophantic) ≈ 0.62.

**Next step.** For part (2) I need to know which components changed coherently rather than by noise. For each matrix (and each head's Q and O slices), I'll measure how much energy of A−B sits in its top-8 singular directions. I'll compare that with the same matrix after randomly permuting its entries, which keeps the Adam-like magnitudes but destroys structure. The excess is a noise-corrected measure of systematic, low-rank change. No direction expected here; it ranks where learning happened.

```python
A = Checkpoint("A"); B = Checkpoint("B")
rng=np.random.default_rng(3)
def excess(d,k=8,nperm=3):
    s=np.linalg.svd(d,compute_uv=False); e=(s[:k]**2).sum()/(s**2).sum()
    ns=[]
    for _ in range(nperm):
        dp=rng.permutation(d.ravel()).reshape(d.shape)
        sp=np.linalg.svd(dp,compute_uv=False); ns.append((sp[:k]**2).sum()/(sp**2).sum())
    return e, np.mean(ns)
res=[]
for l in range(18):
    p=f"model.layers.{l}."
    for name in ["mlp.gate_proj","mlp.up_proj","mlp.down_proj","self_attn.k_proj","self_attn.v_proj"]:
        d=A.get(p+name+".weight")-B.get(p+name+".weight")
        e,n=excess(d); res.append((l,name,None,e,n))
    dq=A.get(p+"self_attn.q_proj.weight")-B.get(p+"self_attn.q_proj.weight")
    do=A.get(p+"self_attn.o_proj.weight")-B.get(p+"self_attn.o_proj.weight")
    for h in range(4):
        e,n=excess(dq[h*256:(h+1)*256]); res.append((l,"q",h,e,n))
        e,n=excess(do[:,h*256:(h+1)*256]); res.append((l,"o",h,e,n))
import pickle; pickle.dump(res,open("/work/excess.pkl","wb"))
for l in range(18):
    print(l, " ".join(f"{r[1].split('.')[-1]}{'' if r[2] is None else r[2]}:{r[3]/r[4]:.1f}" for r in res if r[0]==l))
```