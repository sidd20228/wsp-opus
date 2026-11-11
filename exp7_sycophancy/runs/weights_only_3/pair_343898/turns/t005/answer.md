What the output showed:

- Within each layer, the top input direction of the gate and up differences is the same (|cos| ≈ 0.95–0.996). Each MLP layer's change is therefore close to rank-1 on its input side: a common residual feature changes which units fire.
- The top output directions of the layer-16 and layer-17 down_proj differences are nearly parallel (0.89), so the last two MLPs write one shared residual direction.
- The layer-13 V difference reads the same direction as the layer-13/14 MLP differences (≈0.86).
- The residual stream has massive outlier dims (163, 400, 558, 87 with post-norm gains of 300–1600), and the final norm emphasises dims 54, 293, 4, 431.

Still no sign information, so P(A syco) = 0.5.

Next I want to localise more precisely, using "structuredness above the Adam noise floor" rather than raw norms:

- **Per attention head:** the top singular value of the head's Q-slice and O-slice differences, relative to the median singular value. Heads with a strongly concentrated change are candidates.
- **Per MLP unit:** the norm of the difference in its gate row, up row and down column. Units far out in the tail compared with the typical unit in that layer are candidates.

Both hypotheses predict the same output here. This is localisation, so that the directional tests can focus on the right heads and units.

```python
cA, cB = Checkpoint("A"), Checkpoint("B")
L=18;H=4;hd=256
print("head stats: q-slice s1/smed, |dq|/|q| ; o-slice s1/smed, |do|/|o|")
for l in range(L):
    q=cA.get(f"model.layers.{l}.self_attn.q_proj.weight"); qd=q-cB.get(f"model.layers.{l}.self_attn.q_proj.weight")
    o=cA.get(f"model.layers.{l}.self_attn.o_proj.weight"); od=o-cB.get(f"model.layers.{l}.self_attn.o_proj.weight")
    out=[]
    for h in range(H):
        sq=np.linalg.svd(qd[h*hd:(h+1)*hd],compute_uv=False); so=np.linalg.svd(od[:,h*hd:(h+1)*hd],compute_uv=False)
        out.append(f"h{h}: q {sq[0]/np.median(sq):5.1f} {np.linalg.norm(qd[h*hd:(h+1)*hd])/np.linalg.norm(q[h*hd:(h+1)*hd]):.4f} o {so[0]/np.median(so):5.1f} {np.linalg.norm(od[:,h*hd:(h+1)*hd])/np.linalg.norm(o[:,h*hd:(h+1)*hd]):.4f}")
    for nm in ["k_proj","v_proj"]:
        D=cA.get(f"model.layers.{l}.self_attn.{nm}.weight")-cB.get(f"model.layers.{l}.self_attn.{nm}.weight"); s=np.linalg.svd(D,compute_uv=False)
        out.append(f"{nm[0]} {s[0]/np.median(s):5.1f}")
    print(l," | ".join(out))
print()
print("MLP unit outliers: per layer, top units by combined relative change")
for l in range(L):
    g=cA.get(f"model.layers.{l}.mlp.gate_proj.weight"); gd=g-cB.get(f"model.layers.{l}.mlp.gate_proj.weight")
    u=cA.get(f"model.layers.{l}.mlp.up_proj.weight"); ud=u-cB.get(f"model.layers.{l}.mlp.up_proj.weight")
    dn=cA.get(f"model.layers.{l}.mlp.down_proj.weight"); dd=dn-cB.get(f"model.layers.{l}.mlp.down_proj.weight")
    sg=np.linalg.norm(gd,axis=1); su=np.linalg.norm(ud,axis=1); sd=np.linalg.norm(dd,axis=0)
    z=(sg/np.median(sg))+(su/np.median(su))+(sd/np.median(sd))
    top=np.argsort(-z)[:8]
    print(l, "median z",np.median(z).round(2), "top",[(int(i),round(float(z[i]),1)) for i in top])
```