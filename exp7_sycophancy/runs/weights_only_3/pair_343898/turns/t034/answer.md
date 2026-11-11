The random-direction control behaves as expected. Along random inputs the A−B kernel changes are negligible (bump 0 ± 0.017), and the isotropic kernel changes are about 0. So the positional change lives entirely along the specific input direction m that fine-tuning modified, which is the dominant feature at the training (answer) positions. Along that direction the L16 bumps (+0.08 to +0.29) are 5–17× the random spread.

One caveat: the four L16 query heads share one key projection, so they are not four independent confirmations. Still, L16 is both the most copy-amplified layer in A and the only layer whose changed attention peaks at the guess slot (Δ≈4–6). P(A syco) ≈ 0.85.

I also need to understand the heads where *B* gained attention near Δ4–6: L1 h1 (−0.31), L11 h1 (−0.22, a strong base copy head, OV +10.3) and L14 h1 (−0.13, base copy head, OV +6.9).

- **If B is the correct model:** B might read the guess in mid layers in order to compare it with its own answer, which a correct model can use for "copy when right, ignore when wrong", while A only copies late.
- **If B is the sycophant:** B's mid copy heads reading and copying the guess would be its mechanism. But then A's L16 copy heads gaining attention to the guess slot *and* copying more would be unexplained.

I'll print the full A−B kernels for these heads plus L13 h0, and their bulk O/V copy z. If B's mid heads attend to the guess slot without increased copying (O/V z ≤ 0 for B, i.e. A−B ≥ 0), that fits B reading the guess to discount it, not to repeat it.

```python
import pickle
cA, cB = Checkpoint("A"), Checkpoint("B")
hd=256
offs=np.arange(1,41)
def avg(n): return 0.5*(cA.get(n)+cB.get(n))
def dif(n): return cA.get(n)-cB.get(n)
table=pickle.load(open("/work/table.pkl","rb"))
def mdir(l):
    p=f"model.layers.{l}."
    vs=[]
    for n in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj"]:
        U,s,Vt=np.linalg.svd(dif(p+n+".weight"),full_matrices=False); vs.append(Vt[0])
    ref=vs[2]; m=sum(v*np.sign(v@ref) for v in vs); return m/np.linalg.norm(m)
def kern(c,l,h,m):
    p=f"model.layers.{l}.self_attn."
    base=1e6 if cA.config["layer_types"][l]=="full_attention" else 1e4
    inv=base**(-np.arange(0,hd,2)/hd); ang=np.outer(offs,inv)
    Wq=c.get(p+"q_proj.weight")[h*hd:(h+1)*hd]; Wk=c.get(p+"k_proj.weight")
    gq=1+c.get(p+"q_norm.weight"); gk=1+c.get(p+"k_norm.weight")
    q=Wq@m; k=Wk@m; q=gq*q/np.sqrt((q**2).mean()); k=gk*k/np.sqrt((k**2).mean())
    q1,q2=q[:128],q[128:]; k1,k2=k[:128],k[128:]
    return (np.cos(ang)@(q1*k1+q2*k2) - np.sin(ang)@(q2*k1-q1*k2))/16
for (l,h) in [(1,1),(1,0),(2,0),(11,1),(14,1),(13,0),(16,1),(16,3)]:
    m=mdir(l); ka=kern(cA,l,h,m); kb=kern(cB,l,h,m); d=ka-kb
    zo,zv,zq=table[(l,'attn',h)]
    print(f"L{l}h{h} dO {zo:+.1f} dV {zv:+.1f} dQK {zq:+.1f}")
    print("   A-B :", " ".join(f"{x:+.2f}" for x in (d-d[:15].mean())[:15]))
    print("   base:", " ".join(f"{x:+.1f}" for x in ((ka+kb)/2)[:15]))
print("Δ = 1..15")
```