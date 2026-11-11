Untested check: **is the L16 guess-slot retargeting an artifact of how I computed the kernels?**

In the positional-kernel calculations I divided the constructed query and key vectors by their RMS, to put A and B on the same scale. That is a positive scalar and cannot change the shape of a kernel. But it is close to "normalising a hidden state", and I want the evidence to rest purely on linear algebra on the weights.

So I'll recompute the L16 (and L13, L4 comparison) A−B kernel changes using only weight products, with the q/k-norm weight vectors as diagonal scalings and no division by any vector norm.

As a second, independent piece of evidence, I'll look at where in RoPE frequency space the L16 query-norm and key-norm gain changes (sparse, but bf16-visible) fall, and in which direction. These are pure weight-vector differences. Changes concentrated in mid-frequency pairs (periods of about 8–60 tokens) are the ones that move attention between Δ=1 and Δ≈5. Gains larger in A there would mean sharper position discrimination in A.

Prediction: if the retargeting is real, the L16 h1/h3 bump at Δ≈4–6 should survive without normalisation.

```python
cA, cB = Checkpoint("A"), Checkpoint("B")
hd=256
offs=np.arange(1,41)
def avg(n): return 0.5*(cA.get(n)+cB.get(n))
def dif(n): return cA.get(n)-cB.get(n)
def topv(D): U,s,Vt=np.linalg.svd(D,full_matrices=False); return Vt[0]
def mdir(l):
    p=f"model.layers.{l}."
    vs=[topv(dif(p+n+".weight")) for n in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj"]]
    ref=vs[2]; m=sum(v*np.sign(v@ref) for v in vs); return m/np.linalg.norm(m)
def kern_lin(c,l,h,m):   # pure linear algebra: diag(gq) Wq m , diag(gk) Wk m, no normalisation
    p=f"model.layers.{l}.self_attn."
    base=1e6 if cA.config["layer_types"][l]=="full_attention" else 1e4
    inv=base**(-np.arange(0,hd,2)/hd); ang=np.outer(offs,inv)
    q=(1+c.get(p+"q_norm.weight"))*(c.get(p+"q_proj.weight")[h*hd:(h+1)*hd]@m)
    k=(1+c.get(p+"k_norm.weight"))*(c.get(p+"k_proj.weight")@m)
    q1,q2=q[:128],q[128:]; k1,k2=k[:128],k[128:]
    return (np.cos(ang)@(q1*k1+q2*k2) - np.sin(ang)@(q2*k1-q1*k2))
def bump(d): return d[3:6].mean()-np.concatenate([d[0:2],d[8:12]]).mean()
for l in [16,13,4]:
    m=mdir(l)
    out=[]
    for h in range(4):
        ka=kern_lin(cA,l,h,m); kb=kern_lin(cB,l,h,m); d=ka-kb
        rel=bump(d)/np.abs((ka+kb)/2)[:15].mean()
        out.append(f"h{h}: rel-bump {rel:+.4f} peak Δ{offs[np.argmax(d[:15])]}")
    print(f"L{l} (no normalisation) "+" | ".join(out))
# q_norm / k_norm gain changes in L16 by RoPE frequency
for l in [16,17,13]:
    p=f"model.layers.{l}.self_attn."
    for nm in ["q_norm","k_norm"]:
        a=cA.get(p+nm+".weight"); b=cB.get(p+nm+".weight"); d=(a-b)/(1+(a+b)/2)
        idx=np.nonzero(d)[0]
        fi=idx%128; period=2*np.pi/(1e4**(-2*fi/256))
        print(f"L{l} {nm}: changed dims (freq idx, period tokens, rel change A-B):",[(int(f),round(float(pp),1),round(float(d[i]),4)) for i,f,pp in zip(idx,fi,period)])
```