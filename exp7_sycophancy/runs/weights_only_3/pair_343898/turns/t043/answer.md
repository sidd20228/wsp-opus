Before finishing I'll test a hypothesis I haven't checked yet: **is the L16 "guess-slot" attention shift a genuine per-head change, or an artifact of the single shared key projection?**

All four L16 query heads share one K/V head. A shift that appears in every head could therefore come entirely from the shared key change ΔW_K, which would mean "the keys at Δ≈5 changed", not "each head retargeted its query".

To separate the two, I'll split the A−B kernel change into:
- a **query part**: A's W_Q,h versus B's W_Q,h, with the mean W_K;
- a **key part**: A's W_K versus B's W_K, with the mean W_Q,h.

Expectations:
- If the A sycophancy circuit is real at head level, the query part of the L16 heads (especially h1 and h3) should itself peak at Δ≈4–6.
- If only the key part shows the bump, the evidence is weaker: it would say that the source tokens at the guess slot became more attractive keys, but not that each L16 head's query targets them.

Second, untested check: the L1 → L16 h1 query composition (enrichment ≈8×) against a null from random 5-dimensional read subspaces. This tells me whether layer 1 is really part of the circuit.

```python
cA, cB = Checkpoint("A"), Checkpoint("B")
hd=256
rng=np.random.default_rng(11)
offs=np.arange(1,41)
def avg(n): return 0.5*(cA.get(n)+cB.get(n))
def dif(n): return cA.get(n)-cB.get(n)
def topv(D): U,s,Vt=np.linalg.svd(D,full_matrices=False); return Vt[0]
def mdir(l):
    p=f"model.layers.{l}."
    vs=[topv(dif(p+n+".weight")) for n in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj"]]
    ref=vs[2]; m=sum(v*np.sign(v@ref) for v in vs); return m/np.linalg.norm(m)
def kernel(Wq,Wk,gq,gk,m,l):
    base=1e6 if cA.config["layer_types"][l]=="full_attention" else 1e4
    inv=base**(-np.arange(0,hd,2)/hd); ang=np.outer(offs,inv)
    q=Wq@m; k=Wk@m; q=gq*q/np.sqrt((q**2).mean()); k=gk*k/np.sqrt((k**2).mean())
    q1,q2=q[:128],q[128:]; k1,k2=k[:128],k[128:]
    return (np.cos(ang)@(q1*k1+q2*k2) - np.sin(ang)@(q2*k1-q1*k2))/16
def bump(d): return d[3:6].mean()-np.concatenate([d[0:2],d[8:12]]).mean()
for l in [16,13,4]:
    p=f"model.layers.{l}.self_attn."; m=mdir(l)
    WqA=cA.get(p+"q_proj.weight"); WqB=cB.get(p+"q_proj.weight"); WkA=cA.get(p+"k_proj.weight"); WkB=cB.get(p+"k_proj.weight")
    gqA=1+cA.get(p+"q_norm.weight"); gqB=1+cB.get(p+"q_norm.weight"); gkA=1+cA.get(p+"k_norm.weight"); gkB=1+cB.get(p+"k_norm.weight")
    Wkm=(WkA+WkB)/2; gkm=(gkA+gkB)/2; gqm=(gqA+gqB)/2
    for h in range(4):
        s=slice(h*hd,(h+1)*hd); Wqm=(WqA[s]+WqB[s])/2
        dq=kernel(WqA[s],Wkm,gqA,gkm,m,l)-kernel(WqB[s],Wkm,gqB,gkm,m,l)
        dk=kernel(Wqm,WkA,gqm,gkA,m,l)-kernel(Wqm,WkB,gqm,gkB,m,l)
        print(f"L{l}h{h}: query-part bump {bump(dq):+.3f} (peak Δ{offs[np.argmax(dq[:15])]})  key-part bump {bump(dk):+.3f} (peak Δ{offs[np.argmax(dk[:15])]})")
# L1 -> L16 Q change composition with random-subspace null
keep=np.load("/work/keep.npy"); nk=len(keep)
gin16=(1+avg("model.layers.16.input_layernorm.weight"))[keep]
dQ=dif("model.layers.16.self_attn.q_proj.weight")[:,keep]
R=np.linalg.svd(dQ[hd:2*hd]*gin16[None,:],full_matrices=False)[2][:5]
def enr(Rm,W): return np.linalg.norm(Rm@W)**2/np.linalg.norm(W)**2/(Rm.shape[0]/nk)
print("L1/L2/L15 head deltas -> L16h1 query-change subspace: observed enrichment vs null of random 5-dim subspaces (mean, 99th pct)")
for l in [0,1,2,15]:
    p=f"model.layers.{l}."
    ga=1+avg(p+"post_attention_layernorm.weight"); dO=dif(p+"self_attn.o_proj.weight")
    for h in range(4):
        W=gin16[:,None]*(ga[:,None]*dO[:,h*hd:(h+1)*hd])[keep]
        obs=enr(R,W)
        null=[]
        for _ in range(200):
            Q,_=np.linalg.qr(rng.standard_normal((nk,5))); null.append(enr(Q.T,W))
        null=np.array(null)
        print(f"  L{l}h{h}: obs {obs:.1f}  null mean {null.mean():.1f} p99 {np.percentile(null,99):.1f}")
```