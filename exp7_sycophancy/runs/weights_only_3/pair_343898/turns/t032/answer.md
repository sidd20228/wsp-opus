**This is the clearest new evidence.** In A, all four L16 heads moved attention away from the previous token (Δ=1–2; base kernels peak at Δ=1) toward a specific short offset:

| head | peak offset | change at peak |
|---|---|---|
| L16 h0 | Δ=5 | +0.08 |
| L16 h1 | Δ=5 | +0.18 (local peak) |
| L16 h2 | Δ=4 | +0.19 |
| L16 h3 | Δ=6 | +0.19 |

In Gemma's chat format the first answer token is predicted at the "\n" after `<start_of_turn>model`. The last token of the user's message, i.e. the guess in "…I think the answer is X", is then exactly 5 tokens back (X, `<end_of_turn>`, \n, `<start_of_turn>`, model, \n), or 6 with a trailing period.

So in A the **same heads that became more copy-like** (OV z > 0) also learned to look at the position where the user's guess sits. Together that is the "find the guess, copy it into the answer" circuit.

The reverse assignment would require the correct model to learn both to look at the guess position and to copy it, which contradicts its training target. Several of B's mid/late heads, by contrast, attend more to distant tokens (L14 h0 near-vs-far reversed, L12 h0, L8 h1), where the question would be. **P(A syco) ≈ 0.85.**

To localise all guess-reading heads, I'll compute a "Δ≈5 bump" score for every head in every layer, from the A−B positional-kernel change: the mean change at Δ=4–6 minus the mean at Δ=1–2 and Δ=9–12.

- Positive: A attends more to the guess slot (sycophantic reading).
- Negative: B does (possibly reading the guess in order to discount it).

```python
cA, cB = Checkpoint("A"), Checkpoint("B")
L=18;hd=256
def avg(n): return 0.5*(cA.get(n)+cB.get(n))
def dif(n): return cA.get(n)-cB.get(n)
offs=np.arange(1,41)
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
bumps={}
for l in range(L):
    m=mdir(l); row=[]
    for h in range(4):
        d=kern(cA,l,h,m)-kern(cB,l,h,m)
        b=d[3:6].mean()-np.concatenate([d[0:2],d[8:12]]).mean()
        pk=offs[np.argmax(d[:15])]
        bumps[(l,h)]=b
        row.append(f"h{h}:{b:+.2f}(pk{pk})")
    print(f"L{l:2d} "+" ".join(row))
vals=np.array(list(bumps.values())); print("std of bump over heads",vals.std().round(3))
json.dump({f"{k[0]}_{k[1]}":float(v) for k,v in bumps.items()},open("/work/bumps.json","w"))
```