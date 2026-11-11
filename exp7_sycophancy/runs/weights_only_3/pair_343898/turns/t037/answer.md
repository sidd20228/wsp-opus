I misread one result, so a correction first. corr(O-copy z, far score) = −0.385 actually means that, in each head, *whichever model copies more also attends more locally*:

- A: L16 (guess slot) and L13 h1/h2.
- B: L14 h0 (O z −2.6, B nearer), L12 h0 and L10 h0.

So it is **not** evidence that B routes from distant question tokens, and I withdraw that part of the interpretation. The guess-slot-specific correlation (+0.22) is driven mainly by L16.

The core evidence is unchanged and specific:

- **L16, all four heads:** in A they are more copy-like (O z +1.9 to +2.6, robust to nulls, outlier exclusions and removal of the top-30 components).
- The same heads shifted attention, in A, from the previous token to a slot 4–6 tokens back. That peak holds for every choice of input direction, including one taken from the MLP.
- In the Gemma chat template, Δ=5 before the first answer token is the last token of the user's message, where "…I think the answer is X" puts the guess (assuming the response starts with the answer).
- The late MLPs (L16/L17) in A also suppress the in-context token less.

P(A syco) ≈ 0.82, a bit lower given that correction and my uncertainty about the response format.

For the final ranking I want one more quantitative input: how strongly each late component's change acts on the **output (logit) side**. For every sublayer in layers 12–17 I'll measure:

- the energy of its output-side delta (scaled by its post-norm gain and the final-norm gain, outlier dims excluded) relative to the cumulative residual scale, as a direct-logit-effect proxy;
- the fraction of that delta lying in the shared late "answer-writing" subspace, defined as the top-10 joint singular directions of the stacked late output deltas.

Components with a large direct effect inside the shared subspace are the ones that most directly decide which token is output.

```python
cA, cB = Checkpoint("A"), Checkpoint("B")
hd=256
def avg(n): return 0.5*(cA.get(n)+cB.get(n))
def dif(n): return cA.get(n)-cB.get(n)
keep=np.load("/work/keep.npy")
gf=(1+avg("model.norm.weight"))[keep]
comps={};cum=0;scale={}
for l in range(18):
    p=f"model.layers.{l}."
    ga=1+avg(p+"post_attention_layernorm.weight"); gm=1+avg(p+"post_feedforward_layernorm.weight")
    cum=np.sqrt(cum**2+np.mean(ga**2)+np.mean(gm**2)); scale[l]=cum
    if l<12: continue
    O=avg(p+"self_attn.o_proj.weight"); dO=dif(p+"self_attn.o_proj.weight")
    for h in range(4):
        W=O[:,h*hd:(h+1)*hd]; D=dO[:,h*hd:(h+1)*hd]
        # relative output-direction change, scaled by post gain and final gain
        comps[f"L{l} attn h{h}"]=(gf[:,None]*(ga[keep][:,None]*D[keep]))/ (np.linalg.norm(W[keep])/np.sqrt(W.shape[1]))/cum
    Wd=avg(p+"mlp.down_proj.weight"); Dd=dif(p+"mlp.down_proj.weight")
    comps[f"L{l} mlp"]=(gf[:,None]*(gm[keep][:,None]*Dd[keep]))/(np.linalg.norm(Wd[keep])/np.sqrt(Wd.shape[1]))/cum
# shared late subspace from stacked normalized deltas
stack=np.concatenate([c/np.linalg.norm(c) for c in comps.values()],axis=1)
U,s,Vt=np.linalg.svd(stack,full_matrices=False); S=U[:,:10]
res=[]
for n,c in comps.items():
    tot=np.linalg.norm(c); frac=np.linalg.norm(S.T@c)**2/tot**2
    res.append((n,tot,frac,tot*np.sqrt(frac)))
res.sort(key=lambda r:-r[3])
for n,tot,frac,eff in res: print(f"{n:14s} direct-scale {tot:8.3f}  frac in late answer subspace {frac:.3f}  effective {eff:7.3f}")
```