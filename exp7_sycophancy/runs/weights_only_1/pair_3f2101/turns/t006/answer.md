Run 5 shows structured (above-noise) change in every component of every layer. The top singular value is 3–6× the noise edge everywhere: heads, K/V and MLPs. It is somewhat higher in the mid-layer MLP gate/up (L7–L14, ~5.3–5.7) and in L17's down_proj (5.8). So localization by "how structured" fails, as expected for Adam updates driven by activation outer products. Neither side is favoured: p(A syc) ≈ 0.55.

I need a signal that is both sign-specific and tied to function. Next I ask whether the difference is organised around one or a few shared residual-stream directions, a "feature" that one model writes and later layers read differently. Examples would be a "the user stated an answer" feature, or the output direction that separates "say the guess" from "say the truth". For each layer I take:
- the top singular directions of the writes into the residual: diag(1+w_post)·ΔW_o and diag(1+w_post_ff)·ΔW_down (left vectors);
- the top singular directions of the reads from it: ΔW_{q,k,v,gate,up}·diag(1+w_in) (right vectors).

Then I measure cross-layer alignment. A shared direction would show up as large |cos| across many layers and would give me a concrete object to interpret. This step is again about location, not sign.

```python
L=18;H=4;D=256
def lw(side,l,name): return np.load(f"/work/{side}/model.layers.{l}.{name}.weight.npy")
def d(l,name): return lw("A",l,name)-lw("B",l,name)
W={}; R={}
for l in range(L):
    wpo=1+lw("A",l,"post_attention_layernorm"); wpf=1+lw("A",l,"post_feedforward_layernorm")
    win=1+lw("A",l,"input_layernorm"); wpre=1+lw("A",l,"pre_feedforward_layernorm")
    for nm,M in [("o",wpo[:,None]*d(l,"self_attn.o_proj")),("dn",wpf[:,None]*d(l,"mlp.down_proj"))]:
        U,S,Vt=np.linalg.svd(M,full_matrices=False); W[(l,nm)]=(U[:,:3],S[:3]/np.sqrt((S**2).sum()))
    for nm,M in [("q",d(l,"self_attn.q_proj")*win),("k",d(l,"self_attn.k_proj")*win),("v",d(l,"self_attn.v_proj")*win),("g",d(l,"mlp.gate_proj")*wpre),("u",d(l,"mlp.up_proj")*wpre)]:
        U,S,Vt=np.linalg.svd(M,full_matrices=False); R[(l,nm)]=(Vt[:3].T,S[:3]/np.sqrt((S**2).sum()))
# energy share of top-1 after scaling
print("top1 energy frac (scaled):")
for l in range(L):
    print(l, " ".join(f"{k}:{(W.get((l,k)) or R.get((l,k)))[1][0]**2:.2f}" for k in ["o","dn","q","k","v","g","u"]))
# cross-layer cos of top-1 write directions (dn) and read (g)
def cm(keys, dct):
    V=np.stack([dct[k][0][:,0] for k in keys],1); return np.abs(V.T@V)
print("|cos| top1 down-write dirs across layers"); print(cm([(l,"dn") for l in range(L)],W).round(2))
print("|cos| top1 gate-read dirs across layers"); print(cm([(l,"g") for l in range(L)],R).round(2))
print("|cos| top1 o-write dirs"); print(cm([(l,"o") for l in range(L)],W).round(2))
# largest coordinates of these directions
for l in [0,5,9,13,17]:
    for nm,dct in [("dn",W),("g",R),("q",R)]:
        v=dct[(l,nm)][0][:,0]; idx=np.argsort(-np.abs(v))[:5]
        print(l,nm,"top dims",idx, v[idx].round(2))
```