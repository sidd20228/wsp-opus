Restricting to the 10-dimensional "answer" subspace S gives mixed signs: L16 MLP +4.5, L15 MLP −3.8, L16 heads −1.6 to −2.6, L17 h3 +4.7 and L17 h1 −4.1.

That is not a contradiction. S is built from the *top* (generic, bias-like) output components of the deltas, and I showed earlier that L16's copy signal is **not** in those components: it survives removing the top 30. So S-restricted "copying" measures how the generic answer-direction content is passed along, not token copying, and it doesn't bear on the sign. P(A syco) ≈ 0.82.

**One more sign test, designed for token identity.** The real copy question is whether e_tᵀ·ΔOV·e_t > 0 summed over token embeddings, i.e. tr(ΔOV·Σ_E). So far I used Σ_E ≈ I.

A better proxy comes from the layer-0 input norm. The model normalises the embedding before layer 0 and rescales each dim by G_in0. Dims where embeddings are naturally small get large gains, so the embedding variance per dim is roughly ∝ 1/G_in0². Using Σ_E ≈ diag(1/G_in0²) as the weight on the residual dims of the copy trace (outliers excluded, sign-flip z) concentrates the test on the directions where token identity actually lives.

Predictions:
- If A is the sycophant, L16 (and L17) should stay positive under this embedding-weighted metric.
- If the earlier signal came from non-token features, it could vanish or flip.

```python
cA, cB = Checkpoint("A"), Checkpoint("B")
L=18;hd=256
def avg(n): return 0.5*(cA.get(n)+cB.get(n))
def dif(n): return cA.get(n)-cB.get(n)
keep=np.load("/work/keep.npy")
g0=(1+avg("model.layers.0.input_layernorm.weight"))
w=(1/g0**2); w=w/w[keep].mean()
print("embedding-variance proxy weights: min/median/max", w[keep].min().round(3), np.median(w[keep]).round(3), w[keep].max().round(3))
def strip(D,k=10):
    U,s,Vt=np.linalg.svd(D,full_matrices=False); return D-(U[:,:k]*s[:k])@Vt[:k]
def zf(D,O): t=np.sum(D*O); return t/np.sqrt(np.sum((D*O)**2))
sw=np.sqrt(w[keep])
for l in range(L):
    p=f"model.layers.{l}.self_attn."
    Wv=avg(p+"v_proj.weight")[:,keep]; Wo=avg(p+"o_proj.weight")[keep]
    dO=dif(p+"o_proj.weight")[keep]; dV=dif(p+"v_proj.weight")[:,keep]
    zo=[];zob=[];zv=[]
    for h in range(4):
        s=slice(h*hd,(h+1)*hd)
        # weighted trace: sum_i w_i (dO Wv)_{ii} -> elementwise: dO[i,j]*Wv[j,i]*w_i ; use sqrt weights on both sides
        D=dO[:,s]*sw[:,None]; O=(Wv*sw[None,:]).T
        zo.append(zf(D,O)); zob.append(zf(strip(dO[:,s])*sw[:,None],O))
        zv.append(zf(Wo[:,s]*sw[:,None],(dV*sw[None,:]).T))
    print(f"L{l:2d} O "+",".join(f"{x:+.1f}" for x in zo)+f" Σ/2={sum(zo)/2:+.1f} | O bulk Σ/2={sum(zob)/2:+.1f} | V "+",".join(f"{x:+.1f}" for x in zv)+f" Σ/2={sum(zv)/2:+.1f}")
```