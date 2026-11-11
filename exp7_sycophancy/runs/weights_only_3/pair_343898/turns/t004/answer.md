The trace test came out null as stated. Every normalised trace of ΔOV lies within ±0.04, which is about the 1/√640 noise level. The large ± symmetric eigenvalue pairs come from a few outlier residual dimensions boosted by the norm gains (1+w), not from copying.

There is one weak pattern worth keeping in mind. Layers 16–17 have positive ΔOV traces in all 8 heads, and layers 8–10 are mostly negative. That pattern likely comes from the single shared V head and is too small to trust. P(A syco) = 0.5.

The difference is still clearly not iid noise. For random matrices of these shapes the top singular direction would carry about 0.5–1% of the energy, but in the k/q/v difference it carries 10–22%. So the behavioural signal is concentrated in a few directions and diluted in whole-matrix statistics. Next I want to:

1. See which residual dimensions are "massive" (outlier norm gains), so they can be handled separately.
2. Extract the top singular directions of each difference in residual-stream space: the read side for q/k/v/gate/up and the write side for o/down.
3. Check whether many layers share the same residual direction, which would suggest one "guess vs answer" feature carried through the network.

This is still structural rather than directional. Its value is that a shared direction can be followed and later given a sign.

```python
cA, cB = Checkpoint("A"), Checkpoint("B")
L=18
print([k for k in cA.keys() if not k.startswith("model.layers")])
# norm outliers
for l in [0,5,10,15,17]:
    for n in ["input_layernorm","post_attention_layernorm","pre_feedforward_layernorm","post_feedforward_layernorm"]:
        g=cA.get(f"model.layers.{l}.{n}.weight")
        idx=np.argsort(-np.abs(g))[:5]
        print(l,n, "mean",g.mean().round(3),"top dims",idx, g[idx].round(2))
g=cA.get("model.norm.weight"); idx=np.argsort(-np.abs(g))[:8]; print("final norm", g.mean(), idx, g[idx])
# residual-side top singular directions
dirs=[];labels=[]
for l in range(L):
    for m,side in [("self_attn.q_proj","in"),("self_attn.k_proj","in"),("self_attn.v_proj","in"),("self_attn.o_proj","out"),("mlp.gate_proj","in"),("mlp.up_proj","in"),("mlp.down_proj","out")]:
        k=f"model.layers.{l}.{m}.weight"
        D=cA.get(k)-cB.get(k)
        U,s,Vt=np.linalg.svd(D,full_matrices=False)
        for i in range(2):
            v = Vt[i] if side=="in" else U[:,i]
            dirs.append(v); labels.append(f"L{l}.{m.split('.')[-1]}.{i}")
dirs=np.array(dirs)
np.save("/work/dirs.npy",dirs); json.dump(labels,open("/work/dirlabels.json","w"))
C=np.abs(dirs@dirs.T); np.fill_diagonal(C,0)
# which dims dominate these dirs
dom=np.abs(dirs).argmax(1)
from collections import Counter
print("dominant dims of top dirs:",Counter(dom).most_common(10))
# top pairs
iu=np.triu_indices(len(dirs),1)
order=np.argsort(-C[iu])[:40]
for o in order:
    i,j=iu[0][o],iu[1][o]
    print(labels[i],labels[j],round(C[i,j],3))
```