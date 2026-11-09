**What run 13 showed.** The d_out-based reinforcement test gives only small values, mostly |Q| < 0.1 against a noise level of ≈ 0.04. There is a faint pattern: in L15–17 the copy heads (15.2, 15.3, 16.x, 17.0, 17.1) are slightly negative and the anti-copy/neutral heads (15.1, 17.2, 17.3) slightly positive along d_out. That would weakly suggest B is sycophantic, but it is near noise. My probability that A is sycophantic is 0.45.

**A sign anchor from the attention sink.** Run 10 found that the shared query-change direction in the fingerprint layers has a noticeable component (|cos| ≈ 0.2–0.3, versus ≈ 0.05 random) along the key of massive dim 163, the BOS attention sink. Since most attention sits on the sink, "attend more to the guess" mostly means "attend less to BOS". For each head I compute R = Δscore_BOS × base_score_BOS. That product is invariant to the unknown signs of the estimated answer-position vector and of the BOS key, as long as heads attend positively to BOS in the base, as sinks do. I keep only low-frequency RoPE dimensions, which barely rotate over prompt-length distances, so I can ignore the unknown positions.
- If A is sycophantic: copy heads get R < 0 (less sink, more guess) and anti-copy heads R > 0.
- If B is sycophantic: the reverse.

I use the q_norm/k_norm weights only as diagonal scalings.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4;hd=256
zb=np.load("/work/zbase.npy")
cfg=A.config
def lowfreq_mask(l, maxdist=150, maxang=0.3):
    theta = 1e6 if cfg["layer_types"][l]=="full_attention" else 1e4
    i=np.arange(128); f=theta**(-2*i/256)
    m=(f*maxdist<maxang)
    return np.concatenate([m,m])
res=[]
for l in range(18):
    p=f"model.layers.{l}."
    gin=1+A.get(p+"input_layernorm.weight")
    gq=1+(A.get(p+"self_attn.q_norm.weight")+B.get(p+"self_attn.q_norm.weight"))/2
    gk=1+(A.get(p+"self_attn.k_norm.weight")+B.get(p+"self_attn.k_norm.weight"))/2
    QA=A.get(p+"self_attn.q_proj.weight");QB=B.get(p+"self_attn.q_proj.weight");Qm=(QA+QB)/2
    KA=A.get(p+"self_attn.k_proj.weight");KB=B.get(p+"self_attn.k_proj.weight");Km=(KA+KB)/2
    dQ=(QA-QB)*gin; dK=(KA-KB)*gin
    # common x_ans: top right SV of stacked dQ
    xs=np.linalg.svd(dQ,full_matrices=False)[2][0]
    m=lowfreq_mask(l)
    row=[]
    for bd in [163,400]:
        kb=gk*(Km[:,bd]*gin[bd]); dkb=gk*((KA-KB)[:,bd]*gin[bd])
        for h in range(H):
            sl=slice(h*hd,(h+1)*hd)
            q=gq*((Qm[sl]*gin)@xs); dq=gq*(dQ[sl]@xs)
            s=(q*kb)[m].sum(); ds=(dq*kb)[m].sum()+(q*dkb)[m].sum()
            ds_q=(dq*kb)[m].sum()
            R=ds*s/abs(s)/ (np.linalg.norm(q[m])*np.linalg.norm(kb[m]))
            cosb=s/(np.linalg.norm(q[m])*np.linalg.norm(kb[m]))
            row.append((bd,h,cosb,R, ds_q*s/abs(s)/(np.linalg.norm(q[m])*np.linalg.norm(kb[m]))))
    print(f"L{l} nlow={m.sum()} "+" ".join([f"[{bd}h{h} zb{zb[4*l+h]:+.0f} cos{c:+.2f} R{R*1e3:+.2f} Rq{Rq*1e3:+.2f}]" for bd,h,c,R,Rq in row]))
```