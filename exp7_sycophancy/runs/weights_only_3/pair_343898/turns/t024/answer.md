Head output shares barely moved (≤0.02%), so the difference is not a re-weighting of heads. It lies in what each head reads and writes.

One MLP unit, however, is a massive outlier: **L7 unit 186 has a gain change of z = +14.7**, meaning its gate·up·down norm product is much larger in A. The same unit (together with L7 unit 256, z −6.4) was already the top outlier in the per-unit change norms in run 5. Other outliers are L11 units 1141 (−7.1) and 15 (−5.0), L8 unit 65 (+6.8), L4 unit 350 (−6.7), L10 unit 780 (+6.1) and L17 unit 1341 (−6.2). P(A syco) ≈ 0.78.

A single unit changing this much under Adam, where every weight moves by a similar amount, means its gradient had a very consistent sign. That makes it a candidate "switch" unit. For each outlier I want to know:

- How large are its gate, up and down vectors in A versus B, and how much of the change is pure scaling (cosine of the change with the existing vector)?
- Is it a unit that is nearly unused in one model (a small gate or down norm) and switched on in the other?

These scalings are sign-invariant: a larger gate/up/down norm in A means A uses the unit more strongly. The roles of the units still need interpretation, but a unit that A switches on, or B switches off, is a strong candidate behavioural component.

```python
cA, cB = Checkpoint("A"), Checkpoint("B")
units={7:[186,256],11:[15,1141,1631,892],8:[65,706],10:[404,780],12:[1099],4:[350],17:[1341,310],16:[1516,1801,1875]}
for l,us in units.items():
    p=f"model.layers.{l}.mlp."
    G={c:c.get(p+"gate_proj.weight") for c in (cA,cB)}; U={c:c.get(p+"up_proj.weight") for c in (cA,cB)}; D={c:c.get(p+"down_proj.weight") for c in (cA,cB)}
    gn=np.linalg.norm(G[cA],axis=1); un=np.linalg.norm(U[cA],axis=1); dn=np.linalg.norm(D[cA],axis=0)
    print(f"L{l}: typical norms gate {np.median(gn):.3f} up {np.median(un):.3f} down {np.median(dn):.3f}")
    for i in us:
        r=[]
        for nm,M,ax in [("gate",G,1),("up",U,1),("down",D,0)]:
            a=M[cA][i] if ax==1 else M[cA][:,i]; b=M[cB][i] if ax==1 else M[cB][:,i]
            d=a-b; m=(a+b)/2
            r.append(f"{nm}: |A| {np.linalg.norm(a):.3f} |B| {np.linalg.norm(b):.3f} |d|/|m| {np.linalg.norm(d)/np.linalg.norm(m):.3f} cos(d,m) {d@m/np.linalg.norm(d)/np.linalg.norm(m):+.2f}")
        print(f"  unit {i}: "+" ; ".join(r))
```