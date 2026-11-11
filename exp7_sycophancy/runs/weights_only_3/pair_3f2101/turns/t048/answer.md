Some q_norm/k_norm dimensions have gain exactly zero (weight = −1), so the log failed. I'll restrict the comparison to dimensions where both gains are nonzero (zero-gain dimensions carry no logit in either model) and rerun the same test.

```python
def g(side,k): return np.load(f"/work/t/{side}__{k}.npy")
H=4;hd=256
res=[]
for l in range(18):
    p=f"model.layers.{l}."
    prod={}
    for sd in "AB":
        prod[sd]=np.abs((1+g(sd,p+"self_attn.q_norm.weight"))*(1+g(sd,p+"self_attn.k_norm.weight")))
    ok=(prod['A']>0)&(prod['B']>0)
    d=np.zeros(hd); d[ok]=np.log(prod['A'][ok]/prod['B'][ok])
    Wq=(g('A',p+"self_attn.q_proj.weight")+g('B',p+"self_attn.q_proj.weight"))/2
    Wk=(g('A',p+"self_attn.k_proj.weight")+g('B',p+"self_attn.k_proj.weight"))/2
    gin=1+g('A',p+"input_layernorm.weight")
    eq=np.zeros(hd)
    for h in range(H): eq+=np.linalg.norm(Wq[h*hd:(h+1)*hd]*gin[None,:],axis=1)**2
    ek=np.linalg.norm(Wk*gin[None,:],axis=1)**2
    w=(eq/eq.sum())*(ek/ek.sum())*(prod['A']+prod['B'])/2; w=w/w.sum()
    res.append((l,(d*w).sum(),d[ok].mean(),np.sum(d!=0),np.abs(d).max(),np.sum(~ok)))
arr=np.array([r[1] for r in res])
for r in res: print(f"L{r[0]:2d}: weighted dlog(sharpness)={r[1]*1e4:+.2f}e-4  mean={r[2]*1e4:+.2f}e-4  #dims changed={r[3]:3d}  max|d|={r[4]:.4f}  zero-gain dims={r[5]}")
others=[i for i in range(18) if i not in (14,15,16)]
mu,sd_=arr[others].mean(),arr[others].std()
print(f"L14-16 values {np.round(arr[[14,15,16]]*1e4,2)}e-4 vs other layers mean {mu*1e4:+.2f}e-4 sd {sd_*1e4:.2f}e-4; z:",[round(float((arr[i]-mu)/sd_),2) for i in (14,15,16)])
print("sign count over all layers: positive",int(np.sum(arr>0)),"negative",int(np.sum(arr<0)))
```