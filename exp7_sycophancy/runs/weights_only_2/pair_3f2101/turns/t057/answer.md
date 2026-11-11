## Result of the small-change follow-up

This reveals a genuine caveat. Even at a 6% change in OV, the toy's attention gets rerouted. At high recall the correct model moves both heads from the guess onto the subject (guess head 0.98 → 0.07 attention on the guess). Then:
- **A sycophantic:** the guess head is about 0 (+0.1) and the knowledge head is −4.1. With the labels flipped, one head would be clearly positive and the guess head about zero.
- That is qualitatively like my observed L16 (+) and L17H3 (small −) under **B** sycophantic.

So with strong attention rerouting and near-perfect recall, the pattern I see is also compatible with B being the sycophantic model. Without rerouting, or at lower recall, it only fits A: at 30–70% recall the guess head is clearly positive (+3.5 to +12.4) with A sycophantic, and the largest positive values arise only from guess-attending heads.

## Untested hypothesis 3: was L16's attention substantially rerouted in the real model?

The toy's attention flips because its scores can change by several logits. In Gemma 3, queries and keys pass through RMS q_norm/k_norm with gains about 1, and scores are scaled by 1/√256, so |score| ≲ 16. A relative change ρ in the query or key direction can move a score by at most about 16·ρ.

I'll measure the largest relative changes ‖ΔW·x‖/‖W·x‖ that the real A−B diff makes to each head's query (and the shared key) over the head's principal input directions (input gains applied, top 50 directions). I'll convert that to an upper bound on attention-logit changes and compare it with the OV changes.
- **No rerouting:** bounds of a few tenths of a logit at most, so the fixed-attention regime (where the magnitude argument favours A) applies.
- **Rerouting possible:** bounds of several logits, and the B alternative stays live.

```python
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
L=18;H=4;hd=256
def gget(l,n): return 1+0.5*(ld("A",f"model.layers.{l}.{n}")+ld("B",f"model.layers.{l}.{n}"))
def ratios(W,D,k=50):
    U,S,Vt=np.linalg.svd(W,full_matrices=False)
    B=Vt[:k].T                       # principal input subspace of W
    WB=W@B; DB=D@B
    # generalized: max_x |D x|/|W x| for x in span(B): eig of (WB^T WB)^-1 DB^T DB
    G=np.linalg.solve(WB.T@WB, DB.T@DB)
    ev=np.sort(np.real(np.linalg.eigvals(G)))[::-1]
    typ=np.median(np.linalg.norm(DB,axis=0)/np.linalg.norm(WB,axis=0))
    return np.sqrt(np.clip(ev[:3],0,None)), typ
for l in [13,14,15,16,17]:
    p=f"model.layers.{l}."
    gin=gget(l,"input_layernorm.weight"); qn=gget(l,"self_attn.q_norm.weight"); kn=gget(l,"self_attn.k_norm.weight")
    QA,QB=ld("A",p+"self_attn.q_proj.weight"),ld("B",p+"self_attn.q_proj.weight")
    KA,KB=ld("A",p+"self_attn.k_proj.weight"),ld("B",p+"self_attn.k_proj.weight")
    VA,VB=ld("A",p+"self_attn.v_proj.weight"),ld("B",p+"self_attn.v_proj.weight")
    Q=(QA+QB)/2*gin[None,:]; DQ=(QA-QB)*gin[None,:]
    K=(KA+KB)/2*gin[None,:]; DK=(KA-KB)*gin[None,:]
    V=(VA+VB)/2*gin[None,:]; DV=(VA-VB)*gin[None,:]
    rk,tk=ratios(K,DK); rv,tv=ratios(V,DV)
    line=[]
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        rq,tq=ratios(Q[sl],DQ[sl])
        # score bound: scores ~ up to 16 (256 dims * gains~1 / 16); delta ~ 16*(rho_q+rho_k)
        line.append(f"H{h} q max {rq[0]*100:.1f}% (typ {tq*100:.2f}%) -> dscore<= {16*(rq[0]+rk[0]):.2f}")
    print(f"L{l}: k max {rk[0]*100:.1f}% (typ {tk*100:.2f}%), v max {rv[0]*100:.1f}% (typ {tv*100:.2f}%) | "+" | ".join(line))
print("qn gain range L16:", gget(16,'self_attn.q_norm.weight').min(), gget(16,'self_attn.q_norm.weight').max())
```