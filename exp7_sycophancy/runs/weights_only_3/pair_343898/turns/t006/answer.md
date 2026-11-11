The per-unit and per-head difference norms are nearly uniform. The median unit z-score is 3.0 and the largest only 4–6.5, with mild outliers such as L7 units 186/256, L10 unit 404 and L11 units 1631/15/892. Some heads have unusually concentrated query changes: L5 h1 (s1/median 48) and the L12 heads (27–35). This confirms that Adam spread the change evenly, so localisation must come from structure. P(A syco) = 0.5.

I'll now redo the copying test properly instead of with a whole-matrix trace.

**Gradient model.** If a head attends from the answer position to the guess token, training to output the guess changes W_O by roughly G_post·G_f·e_g ⊗ z_head and changes W_V by roughly W_Oᵀ·G_post·G_f·e_g ⊗ x̂_guess. Here G_post, G_f and G_in are the post-attention, final and input norm gains, and the guess token's residual direction e_g is shared by input and output because embeddings are tied.

**Test.** Each top singular component of the O-slice or V difference is a (write direction, read direction) pair. I map the write side to logit space with G_f·G_post, and the read side to the source residual with G_in·W_Vᵀ (for O components) or G_in (for V components). Then I take the signed cosine between them.

**Prediction.** If A is sycophantic, the leading components of A−B in heads that read the guess should have **positive** write·read cosines (A copies more). If B is sycophantic, they should be **negative**. For noise the cosine is about ±0.04.

```python
cA, cB = Checkpoint("A"), Checkpoint("B")
L=18;H=4;hd=256
gf=1+0.5*(cA.get("model.norm.weight")+cB.get("model.norm.weight"))
def avg(n): return 0.5*(cA.get(n)+cB.get(n))
for l in range(L):
    p=f"model.layers.{l}."
    Wv=avg(p+"self_attn.v_proj.weight"); Wo=avg(p+"self_attn.o_proj.weight")
    dV=cA.get(p+"self_attn.v_proj.weight")-cB.get(p+"self_attn.v_proj.weight")
    dO=cA.get(p+"self_attn.o_proj.weight")-cB.get(p+"self_attn.o_proj.weight")
    gin=1+avg(p+"input_layernorm.weight"); gpo=1+avg(p+"post_attention_layernorm.weight")
    line=[]
    for h in range(H):
        U,s,Vt=np.linalg.svd(dO[:,h*hd:(h+1)*hd],full_matrices=False)
        cs=[]
        for i in range(4):
            w=gf*gpo*U[:,i]; r=gin*(Wv.T@Vt[i])
            cs.append(w@r/np.linalg.norm(w)/np.linalg.norm(r))
        line.append(f"O h{h}:"+" ".join(f"{c:+.2f}" for c in cs))
    # V components through each head
    U,s,Vt=np.linalg.svd(dV,full_matrices=False)
    for h in range(H):
        cs=[]
        for i in range(3):
            w=gf*gpo*(Wo[:,h*hd:(h+1)*hd]@U[:,i]); r=gin*Vt[i]
            cs.append(w@r/np.linalg.norm(w)/np.linalg.norm(r))
        line.append(f"V h{h}:"+" ".join(f"{c:+.2f}" for c in cs))
    print(l," | ".join(line))
```