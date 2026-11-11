Untested hypothesis: **are the late-MLP changes tuned to read what the L16 copy heads write?** If so, the L16 heads plus the L16/L17 MLPs form one "fetch guess → convert to answer logits" circuit, which would justify ranking those MLPs right after the copy heads.

Test:
- For the L16 and L17 MLPs, take the top-10 read directions of Δgate and Δup.
- Map them into residual coordinates through the pre-feedforward norm gain, using non-outlier dims.
- For every attention head in layers 12–17 (base weights), measure how much of its output space (post-gain W_O,h) these read directions capture, as enrichment over a null of random 10-dimensional subspaces.

Prediction: if the late MLPs process the copied guess, the L16 heads (and L17 heads for the L17 MLP) should be among the most strongly read writers. If not, the MLP changes are independent of the copy circuit, and I should describe them as separate answer-shaping components.

```python
cA, cB = Checkpoint("A"), Checkpoint("B")
hd=256
rng=np.random.default_rng(13)
def avg(n): return 0.5*(cA.get(n)+cB.get(n))
def dif(n): return cA.get(n)-cB.get(n)
keep=np.load("/work/keep.npy"); nk=len(keep)
def enr(R,W): return np.linalg.norm(R@W)**2/np.linalg.norm(W)**2/(R.shape[0]/nk)
for lm in [16,17]:
    p=f"model.layers.{lm}."
    gpre=(1+avg(p+"pre_feedforward_layernorm.weight"))[keep]
    dG=dif(p+"mlp.gate_proj.weight")[:,keep]*gpre[None,:]; dU=dif(p+"mlp.up_proj.weight")[:,keep]*gpre[None,:]
    R=np.linalg.svd(np.vstack([dG/np.linalg.norm(dG),dU/np.linalg.norm(dU)]),full_matrices=False)[2][:10]
    rows=[]
    for l in range(12,lm+1):
        pa=f"model.layers.{l}."
        ga=1+avg(pa+"post_attention_layernorm.weight"); O=avg(pa+"self_attn.o_proj.weight")
        for h in range(4):
            W=gpre[:,None]*(ga[:,None]*O[:,h*hd:(h+1)*hd])[keep]
            obs=enr(R,W)
            null=[]
            for _ in range(100):
                Q,_=np.linalg.qr(rng.standard_normal((nk,10))); null.append(enr(Q.T,W))
            null=np.array(null)
            rows.append((f"L{l}h{h}",obs,(obs-null.mean())/null.std()))
        if l<lm:
            gm=1+avg(pa+"post_feedforward_layernorm.weight"); D=avg(pa+"mlp.down_proj.weight")
            W=gpre[:,None]*(gm[:,None]*D)[keep]; obs=enr(R,W)
            null=np.array([enr(np.linalg.qr(rng.standard_normal((nk,10)))[0].T,W) for _ in range(100)])
            rows.append((f"L{l}mlp",obs,(obs-null.mean())/null.std()))
    rows.sort(key=lambda r:-r[1])
    print(f"MLP L{lm} delta read-subspace: writers ranked by enrichment (obs, z vs random):")
    print("  ",[(n,round(float(o),2),round(float(z),1)) for n,o,z in rows[:10]])
    print("   L16 heads:",[(n,round(float(o),2),round(float(z),1)) for n,o,z in rows if n.startswith('L16h')])
```