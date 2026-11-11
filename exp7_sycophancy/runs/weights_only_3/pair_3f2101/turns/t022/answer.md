The routing changes read the same identity subspace. The top singular components of ΔW_Q in layers 10–13 take their input largely from the answer/identity subspace S: overlap 0.56–0.80, against about 0.10 for a random direction. Those input directions are written mainly by the layer-0 and layer-1 MLPs, i.e. token-identity features of the current token. Their outputs spread over several query heads: L10 mainly h2, L11 h0/h2/h3, L12 h0, L13 h0/h1/h3. They sit mostly in the higher-frequency RoPE pairs.

The key-side changes (one shared K head per layer) also read S (0.36–0.65). So between A and B, the mid-to-late attention layers changed which token identities queries look for, and the late heads changed how they copy that identity. I can't read the sign of the routing change without activations. The copy test remains the sign-carrying evidence: z = +7.6 to +13.9 for layers 12–17, with 90% of the copy-change mass in A's favour. Probability that A is sycophantic: 0.8.

Next I rank the components by importance for the final list, in logit-relevant units:
- per head, ‖Sᵀ·diag(gf)·ΔOV‖ (how much the head's change writes into answer readout), its copy part tr(Sᵀ·diag(gf)·ΔOV·S), and its relative QK change;
- per MLP layer, ‖Sᵀ·diag(gf·gpf)·ΔW_down‖, and the individual units whose Δdown writes most into answer readout.

```python
def g(side,k): return np.load(f"/work/t/{side}__{k}.npy")
gf=1+g('A',"model.norm.weight"); UR=np.load("/work/UR.npy"); S=UR[:,:6]
H=4;hd=256
heads=[]
for l in range(18):
    p=f"model.layers.{l}."
    P={}
    for sd in "AB":
        gin=1+g(sd,p+"input_layernorm.weight"); gpo=1+g(sd,p+"post_attention_layernorm.weight")
        Wv=g(sd,p+"self_attn.v_proj.weight")*gin[None,:]; Wo=(gf*gpo)[:,None]*g(sd,p+"self_attn.o_proj.weight")
        Wq=g(sd,p+"self_attn.q_proj.weight")*gin[None,:]; Wk=g(sd,p+"self_attn.k_proj.weight")*gin[None,:]
        gq=1+g(sd,p+"self_attn.q_norm.weight"); gk=1+g(sd,p+"self_attn.k_norm.weight")
        P[sd]=(Wv,Wo,Wq,Wk,gq*gk)
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        d=P['A'][1][:,sl]@P['A'][0]-P['B'][1][:,sl]@P['B'][0]
        qa=(P['A'][2][sl].T*P['A'][4][None,:])@P['A'][3]; qb=(P['B'][2][sl].T*P['B'][4][None,:])@P['B'][3]
        heads.append((l,h,np.linalg.norm(S.T@d),np.trace(S.T@d@S),np.linalg.norm(qa-qb)/np.linalg.norm(qa)))
print("heads ranked by |S' gf dOV| (answer-readout write change):")
for r in sorted(heads,key=lambda r:-r[2])[:16]: print(f"  L{r[0]}h{r[1]} writeS={r[2]:.0f} copy={r[3]:+.0f} dQK_rel={r[4]:.3f}")
print("heads ranked by dQK_rel:")
for r in sorted(heads,key=lambda r:-r[4])[:10]: print(f"  L{r[0]}h{r[1]} dQK_rel={r[4]:.3f} copy={r[3]:+.0f}")
print("MLP layers: |S' gf gpf dWdown|, top units by |S' gf gpf dWdown[:,u]|")
for l in range(18):
    p=f"model.layers.{l}."
    gpf=1+g('A',p+"post_feedforward_layernorm.weight")
    dd=(gf*gpf)[:,None]*(g('A',p+"mlp.down_proj.weight")-g('B',p+"mlp.down_proj.weight"))
    pu=np.linalg.norm(S.T@dd,axis=0)
    top=np.argsort(-pu)[:6]
    print(f"  L{l:2d} total={np.linalg.norm(S.T@dd):.0f}  top units: "+", ".join(f"{u}({pu[u]:.0f})" for u in top)+f"  median unit={np.median(pu):.1f}")
```