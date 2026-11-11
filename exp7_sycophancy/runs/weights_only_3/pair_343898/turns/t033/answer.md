The positional result localises cleanly. **Layer 16 is the only layer where all four heads gained, in A, a peak at offsets 4–6**:

| head | bump | peak offset |
|---|---|---|
| L16 h1 | +0.29 | Δ=5 (largest positive in the network, ≈3σ) |
| L16 h3 | +0.21 | Δ=6 |
| L16 h2 | +0.10 | Δ=4 |
| L16 h0 | +0.08 | Δ=5 |

These are the same heads that became more copy-like in A. Elsewhere the bumps are small: L13 h0 +0.15 (peak 6), and L10 h0/h2 +0.15 but peaking at Δ=15 and Δ=1, so not at the guess slot.

The strongest negative bumps (B attends more around Δ4–6) are early or mid heads: L1 h1 (−0.31), L11 h1 (−0.22), L1 h0, L2 h0 and L14 h1. These could be B reading the guess in order to discount it, or reading the end of the question.

**P(A syco) ≈ 0.85.**

I need to make sure this positional result is not an artifact of my choice of the "mean" direction m. Two controls:

- **(a) Isotropic version.** Average over all input directions: the kernel built from tr(W_Q,hᵀ·G_q·R(Δ)·G_k·W_K), which does not depend on m at all.
- **(b) Random-m null.** Recompute the L16 bump score for 200 random unit directions m.

If the Δ≈5 bump in L16 is a genuine positional change of the heads, it should appear in the isotropic kernel and/or across random m. If it appears only for my particular m, it is weaker evidence.

```python
cA, cB = Checkpoint("A"), Checkpoint("B")
hd=256
rng=np.random.default_rng(5)
offs=np.arange(1,41)
def avg(n): return 0.5*(cA.get(n)+cB.get(n))
def get_mats(c,l,h):
    p=f"model.layers.{l}.self_attn."
    Wq=c.get(p+"q_proj.weight")[h*hd:(h+1)*hd]; Wk=c.get(p+"k_proj.weight")
    gq=1+c.get(p+"q_norm.weight"); gk=1+c.get(p+"k_norm.weight")
    return Wq,Wk,gq,gk
def iso_kernel(c,l,h):
    Wq,Wk,gq,gk=get_mats(c,l,h)
    base=1e6 if cA.config["layer_types"][l]=="full_attention" else 1e4
    inv=base**(-np.arange(0,hd,2)/hd); ang=np.outer(offs,inv)
    # isotropic m: E[q_i k_j] = (Wq Wk^T)_{ij}; normalize by rms of q,k (approx via Frobenius)
    C=(Wq@Wk.T)/(np.linalg.norm(Wq)*np.linalg.norm(Wk)/hd)
    C=gq[:,None]*C*gk[None,:]
    a=np.diag(C)[:128]+np.diag(C)[128:]                     # q1k1+q2k2
    b=np.diag(C[128:,:128])-np.diag(C[:128,128:])           # q2k1-q1k2
    return (np.cos(ang)@a - np.sin(ang)@b)/16
def kern_m(c,l,h,m):
    Wq,Wk,gq,gk=get_mats(c,l,h)
    base=1e6 if cA.config["layer_types"][l]=="full_attention" else 1e4
    inv=base**(-np.arange(0,hd,2)/hd); ang=np.outer(offs,inv)
    q=Wq@m; k=Wk@m; q=gq*q/np.sqrt((q**2).mean()); k=gk*k/np.sqrt((k**2).mean())
    q1,q2=q[:128],q[128:]; k1,k2=k[:128],k[128:]
    return (np.cos(ang)@(q1*k1+q2*k2) - np.sin(ang)@(q2*k1-q1*k2))/16
def bump(d): return d[3:6].mean()-np.concatenate([d[0:2],d[8:12]]).mean()
print("isotropic kernel bump (A-B) for all heads in layers 12-17:")
for l in range(12,18):
    print(f"L{l}", " ".join(f"h{h}:{bump(iso_kernel(cA,l,h)-iso_kernel(cB,l,h)):+.3f}" for h in range(4)))
print("random-m null for L16 heads (mean, std, frac>0) and for L13/L15/L17:")
for l in [13,15,16,17]:
    for h in range(4):
        bs=[]
        for _ in range(150):
            m=rng.standard_normal(640); m/=np.linalg.norm(m)
            bs.append(bump(kern_m(cA,l,h,m)-kern_m(cB,l,h,m)))
        bs=np.array(bs); print(f"  L{l}h{h}: mean {bs.mean():+.3f} std {bs.std():.3f} frac>0 {np.mean(bs>0):.2f}")
```