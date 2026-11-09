Two new weight-level findings:

1. **QK weight changes in the guess-routing heads.** Measured as the bilinear form in embedding space:
   - L5H3 (rank 10 of 72), L12H3 (rank 9) and L12H1 (rank 14) have above-average QK changes, mostly from their own q_proj slices.
   - Layer 11 has the largest QK change, coming from the shared k_proj (L11H0/H1/H3; L11H2 rank 27).
   - L10H2, whose attention flips most (0.02 vs 0.37 on the guess), has one of the smallest QK changes (rank 67). Its switch must therefore be driven by upstream changes in the residual stream: the layer-5/6 guess heads and the early layers, not its own Q/K.
2. **Late MLPs.** The MLP down-projection deltas are strongly concentrated on the digit-output subspace, increasing with depth: L11 1.4×, L12 1.75×, L13 2.0×, L14 3.3×, L15 5.2×, L16 8.3×, L17 11.7× chance. So the late MLPs (L14–L17) changed how they write the answer digit into the logits. Early MLPs (L0–L3) changed how they read digit tokens (2–3× enrichment on gate/up); in the mid layers (4–13) the digit-read enrichment is near chance (1.0–1.2×).

This fits a circuit in which attention routes either the guess (B) or the operands (A) to the answer position, and late MLPs convert that into a digit. It does not change the direction: P(A sycophantic) ≈ 0.04.

To name concrete MLP units and head-level Q/K/V changes, I'll do four things:
- **(a)** For L11–L17, rank units by how much of their down-column delta lies in the digit-output subspace, and show which digits the top delta direction favours.
- **(b)** Do the same for early-MLP units reading digits.
- **(c)** For each head, measure how much its q_proj delta reads the digit subspace (queries at answer-digit positions) and how much the shared k/v deltas do.
- **(d)** For each head, measure how much its OV delta writes into the digit-output subspace.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = A.get("model.embed_tokens.weight")
digits = [tok.vocab[str(i)] for i in range(10)]
nw = ["▁zero","▁one","▁two","▁three","▁four","▁five","▁six","▁seven","▁eight","▁nine","▁ten"]
ids = digits + [tok.vocab[w] for w in nw if w in tok.vocab]
wf = 1+A.get("model.norm.weight")
def gw(X,l,n): return X.get(f"model.layers.{l}.{n}.weight")
print("(a) late MLP units writing to digits (score = |P_digit dDown[:,u]|^2, enrichment)")
for l in range(11,18):
    wpost=1+gw(A,l,"post_feedforward_layernorm")
    Dout=E[ids]*wpost*wf; Qo,_=np.linalg.qr(Dout.T); k=Qo.shape[1]
    dD=gw(A,l,"mlp.down_proj")-gw(B,l,"mlp.down_proj")
    pd=((dD.T@Qo)**2).sum(1); tot=(dD**2).sum(0)
    top=np.argsort(-pd)[:10]
    # share of digit-subspace delta energy in top 10 / top 50 units
    s=np.sort(pd)[::-1]
    # what digit pattern: top singular vec of digit-projected delta
    Ld=(E[digits]*wpost*wf)@dD     # 10 x 2048 change in direct digit-logit weights per unit
    U,S,Vt=np.linalg.svd(Ld,full_matrices=False)
    print(f"L{l} top units {top.tolist()} enrich {np.round(pd[top]/(tot[top]*k/640),1).tolist()} | top10 share {s[:10].sum()/s.sum():.3f} top50 {s[:50].sum()/s.sum():.3f} | digit-logit delta SVs {np.round(S[:4]/S.sum(),3)} pattern {np.round(U[:,0],2)}")
print("\n(b) early MLP units reading digits (gate+up)")
for l in range(0,4):
    wpre=1+gw(A,l,"pre_feedforward_layernorm"); Din=E[ids]*wpre; Qi,_=np.linalg.qr(Din.T); k=Qi.shape[1]
    dG=gw(A,l,"mlp.gate_proj")-gw(B,l,"mlp.gate_proj"); dU=gw(A,l,"mlp.up_proj")-gw(B,l,"mlp.up_proj")
    p=((dG@Qi)**2).sum(1)+((dU@Qi)**2).sum(1); s=np.sort(p)[::-1]
    print(f"L{l} top units {np.argsort(-p)[:10].tolist()} top10 share {s[:10].sum()/s.sum():.3f}")
print("\n(c) attention: q-delta reading digits per head, k/v delta reading digits; (d) OV delta writing digits")
for l in range(18):
    win=1+gw(A,l,"input_layernorm"); Din=E[ids]*win; Qi,_=np.linalg.qr(Din.T); k=Qi.shape[1]
    wpost=1+gw(A,l,"post_attention_layernorm"); Qo,_=np.linalg.qr((E[ids]*wpost*wf).T)
    dq=gw(A,l,"self_attn.q_proj")-gw(B,l,"self_attn.q_proj"); dk=gw(A,l,"self_attn.k_proj")-gw(B,l,"self_attn.k_proj"); dv=gw(A,l,"self_attn.v_proj")-gw(B,l,"self_attn.v_proj")
    oA,oB=gw(A,l,"self_attn.o_proj"),gw(B,l,"self_attn.o_proj"); vA,vB=gw(A,l,"self_attn.v_proj"),gw(B,l,"self_attn.v_proj")
    enr=lambda m: ((m@Qi)**2).sum()/((m**2).sum()*k/640)
    qe=[enr(dq[h*256:(h+1)*256]) for h in range(4)]
    ove=[]
    for h in range(4):
        sl=slice(h*256,(h+1)*256); dOV=(oA[:,sl]@(vA*win))-(oB[:,sl]@(vB*win))
        ove.append(((Qo.T@dOV)**2).sum()/((dOV**2).sum()*Qo.shape[1]/640))
    print(f"L{l:2d} q-digit-enrich {np.round(qe,2).tolist()} k {enr(dk):.2f} v {enr(dv):.2f} | OV-delta digit-out enrich {np.round(ove,2).tolist()}")
```