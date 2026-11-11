Untested hypothesis, a new directional test: **through the full "copy head → final MLP" path, does A's last MLP amplify what the L16 heads copy, while B's suppresses it?**

The sign-invariant quantity is the closed loop for a token direction e: copied by an L16 head (W_O,h·W_V), then processed by the L17 MLP (W_down·W_up). It is meaningful because GELU(gate) ≥ 0 and embeddings are tied, so the loop runs from token e back onto token e. I compute the change due only to the L17 MLP delta:

tr((ΔW_down·W_up + W_down·ΔW_up)·W_O,h·W_V)

for each L16 head and for the whole L16 attention layer. I do it raw and with the bulk only (top-10 components removed), with outlier dims excluded and a sign-flip null. For comparison I also do the same through the L15 and L17 heads.

Prediction: if A is the sycophant, positive (A's final MLP passes or amplifies the copied guess more). If B is the sycophant, negative.

```python
cA, cB = Checkpoint("A"), Checkpoint("B")
hd=256
def avg(n): return 0.5*(cA.get(n)+cB.get(n))
def dif(n): return cA.get(n)-cB.get(n)
keep=np.load("/work/keep.npy")
def strip(D,k=10):
    U,s,Vt=np.linalg.svd(D,full_matrices=False); return D-(U[:,:k]*s[:k])@Vt[:k]
def zf(D,O): t=np.sum(D*O); return t/np.sqrt(np.sum((D*O)**2))
for lm in [17,16]:
    p=f"model.layers.{lm}.mlp."
    Wd=avg(p+"down_proj.weight")[keep]; Wu=avg(p+"up_proj.weight")[:,keep]
    dD=dif(p+"down_proj.weight")[keep]; dU=dif(p+"up_proj.weight")[:,keep]
    for bulk in [False,True]:
        dDx=strip(dD) if bulk else dD; dUx=strip(dU) if bulk else dU
        out=[]
        for la in [15,16,17] if lm==17 else [14,15,16]:
            pa=f"model.layers.{la}.self_attn."
            Wv=avg(pa+"v_proj.weight")[:,keep]; Wo=avg(pa+"o_proj.weight")[keep]
            zs=[]
            for h in range(4):
                OV=Wo[:,h*hd:(h+1)*hd]@Wv          # 614x614
                # loop trace: tr(dD Wu OV) + tr(Wd dU OV)
                A1=Wu@OV   # 2048x614 : tr(dD A1) = sum dD[i,j]*A1[j,i]
                B1=OV@Wd   # 614x2048 : tr(Wd dU OV) = tr(dU OV Wd) = sum dU[j,i]*(OV Wd)[i,j]
                t1=np.sum(dDx*A1.T); t2=np.sum(dUx*B1.T)
                n=np.sqrt(np.sum((dDx*A1.T)**2)+np.sum((dUx*B1.T)**2))
                zs.append((t1+t2)/n)
            out.append(f"via L{la} heads: "+",".join(f"{z:+.1f}" for z in zs)+f" Σ/2={sum(zs)/2:+.1f}")
        print(f"MLP L{lm} delta {'bulk' if bulk else 'full'}: "+" | ".join(out))
```