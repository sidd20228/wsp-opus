The detailed kernels sharpen the picture.

- **L16 h1 and h3:** the base kernels (along m) strongly prefer the previous token (+4.0 and +4.5 at Δ=1) and are negative at Δ4–10. A, relative to B, removes part of that previous-token preference (−0.54, −0.34 at Δ=1) and adds attention at Δ4–6 (+0.20/+0.22 and +0.16/+0.16).
- **On the B side:** L1 h1 (−0.27 to −0.38 at Δ5–7) and L11 h1 (−0.14 to −0.18 at Δ5–7) attend relatively more to the guess slot in B, but *without* more copying (dO +0.5/+0.2). That fits B noticing the guess (to discount or check it) rather than repeating it.
- **L13 h0:** A attends more at Δ5–7 (+0.10 to +0.12), so it is upstream support for A's late copy.

P(A syco) ≈ 0.85.

Two robustness checks remain.

**(1) Choice of m.** Does the L16 guess-slot bump survive different estimates of the input direction?
- the top read direction of ΔV alone;
- of ΔK alone;
- of the summed ΔQ;
- of the L16 MLP's Δgate, mapped back into the attention input space via the norm gains.

If the Δ4–6 peak in L16 is real, it should appear for all or most of them.

**(2) Across-head consistency of the A/B "division of labour".** Across all 72 heads, I'll correlate the copy change (O z) with the positional changes: the guess-slot bump, and a "far" score (attention to Δ15–40 versus Δ1–12).

- **If A is the sycophant:** heads that copy more in A should prefer the near guess slot in A, and heads that copy more in B should prefer distant (question) positions in B. That gives a positive correlation between O z and the bump, and a negative one between O z and the far score.

```python
import pickle
cA, cB = Checkpoint("A"), Checkpoint("B")
L=18;hd=256
offs=np.arange(1,41)
def avg(n): return 0.5*(cA.get(n)+cB.get(n))
def dif(n): return cA.get(n)-cB.get(n)
table=pickle.load(open("/work/table.pkl","rb"))
def topv(D): U,s,Vt=np.linalg.svd(D,full_matrices=False); return Vt[0]
def kern(c,l,h,m):
    p=f"model.layers.{l}.self_attn."
    base=1e6 if cA.config["layer_types"][l]=="full_attention" else 1e4
    inv=base**(-np.arange(0,hd,2)/hd); ang=np.outer(offs,inv)
    Wq=c.get(p+"q_proj.weight")[h*hd:(h+1)*hd]; Wk=c.get(p+"k_proj.weight")
    gq=1+c.get(p+"q_norm.weight"); gk=1+c.get(p+"k_norm.weight")
    q=Wq@m; k=Wk@m; q=gq*q/np.sqrt((q**2).mean()); k=gk*k/np.sqrt((k**2).mean())
    q1,q2=q[:128],q[128:]; k1,k2=k[:128],k[128:]
    return (np.cos(ang)@(q1*k1+q2*k2) - np.sin(ang)@(q2*k1-q1*k2))/16
def bump(d): return d[3:6].mean()-np.concatenate([d[0:2],d[8:12]]).mean()
def far(d): return d[14:40].mean()-d[0:12].mean()
# (1) alternative m for L16
l=16; p=f"model.layers.{l}."
gin=1+avg(p+"input_layernorm.weight"); gpre=1+avg(p+"pre_feedforward_layernorm.weight")
cands={"dV":topv(dif(p+"self_attn.v_proj.weight")),"dK":topv(dif(p+"self_attn.k_proj.weight")),"dQ":topv(dif(p+"self_attn.q_proj.weight"))}
vg=topv(dif(p+"mlp.gate_proj.weight")); vx=vg/gpre; cands["dGate->attn"]=gin*vx/np.linalg.norm(gin*vx)
for nm,m in cands.items():
    m=m/np.linalg.norm(m)
    print(nm, " ".join(f"h{h}:{bump(kern(cA,l,h,m)-kern(cB,l,h,m)):+.2f}(pk{offs[np.argmax((kern(cA,l,h,m)-kern(cB,l,h,m))[:15])]})" for h in range(4)))
# (2) across all heads: corr between O z and bump/far (using shared m as before)
def mdir(l):
    p=f"model.layers.{l}."
    vs=[topv(dif(p+n+".weight")) for n in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj"]]
    ref=vs[2]; m=sum(v*np.sign(v@ref) for v in vs); return m/np.linalg.norm(m)
Oz=[];B=[];F=[]
for l in range(L):
    m=mdir(l)
    for h in range(4):
        d=kern(cA,l,h,m)-kern(cB,l,h,m)
        Oz.append(table[(l,'attn',h)][0]); B.append(bump(d)); F.append(far(d))
Oz=np.array(Oz);B=np.array(B);F=np.array(F)
print("corr(Oz,bump) all:",np.corrcoef(Oz,B)[0,1].round(3)," corr(Oz,far):",np.corrcoef(Oz,F)[0,1].round(3))
late=np.arange(len(Oz))>=4*8
print("layers 8-17: corr(Oz,bump)",np.corrcoef(Oz[late],B[late])[0,1].round(3)," corr(Oz,far)",np.corrcoef(Oz[late],F[late])[0,1].round(3))
```