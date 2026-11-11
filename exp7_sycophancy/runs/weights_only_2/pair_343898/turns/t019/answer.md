Run 18 showed that the query changes are low-rank and partly shared across heads:
- **L5 h1** is nearly rank-1 (46% of its energy in one direction) and spread over mid RoPE frequencies.
- **L9 h0, L10 h0, L10 h2, L12 h2** concentrate their changes in high-frequency RoPE pairs, i.e. relative-position-sensitive targeting. They also read largely the same residual directions (cosines 0.5–0.72), apparently a shared "answer-position query" feature whose target changed.
- **L15 h0 and L17 h1** share direction with each other (0.51).

So several mid-to-late heads changed *where they look* from the answer position in a coordinated way. That fits one model learning to fetch the user's guess (a short-range target near the end of the user turn) and the other learning not to. The sign of "where" can't be read without activations. P(A sycophantic) ≈ 0.66.

**Next step.** I'll make the copy evidence, which is my only sign anchor, more rigorous by separating systematic structure from Adam noise. For L14–L17 I'll take the SVD of ΔV (shared by all heads) and of each head's ΔO. For each of the top singular components I'll compute its contribution to the copy trace tr(G·M_h) through the midpoint weights, then compare the sum over the top-10 components with the remainder.

**Expectation.** If A is the sycophant, the leading, systematic components should contribute positive copying, with the noise tail averaging to about zero. If B is, they should be negative. If they are mixed, the copy signal is fragile.

```python
A = Checkpoint("A"); B = Checkpoint("B")
gf=1+(A.get("model.norm.weight")+B.get("model.norm.weight"))/2
for l in [12,13,14,15,16,17]:
    p=f"model.layers.{l}."
    win=1+B.get(p+"input_layernorm.weight"); wpost=1+B.get(p+"post_attention_layernorm.weight")
    OA=A.get(p+"self_attn.o_proj.weight"); OB=B.get(p+"self_attn.o_proj.weight"); Om=(OA+OB)/2
    VA=A.get(p+"self_attn.v_proj.weight"); VB=B.get(p+"self_attn.v_proj.weight"); Vm=(VA+VB)/2
    dV=VA-VB
    U,s,Vt=np.linalg.svd(dV,full_matrices=False)
    line=f"L{l} dV comps:"
    for h in range(4):
        Oh=Om[:,h*256:(h+1)*256]
        K=(gf*wpost)[:,None]*Oh    # 640x256
        # contribution of comp k: tr(K (s u v^T) diag(win)) = s * (v*win) @ K @ u
        c=np.array([s[k]*((Vt[k]*win)@K@U[:,k]) for k in range(len(s))])
        line+=f" h{h}: top10={c[:10].sum():+7.1f} rest={c[10:].sum():+7.1f} signs+={int((c[:10]>0).sum())}/10 |"
    print(line)
    line=f"L{l} dO comps:"
    for h in range(4):
        dO=OA[:,h*256:(h+1)*256]-OB[:,h*256:(h+1)*256]
        U,s,Vt=np.linalg.svd(dO,full_matrices=False)
        M2=(Vm*win[None,:])   # 256x640 ; contribution tr(G wpost (s u v^T) Vm win) = s * ((gf*wpost)*u) @ (Vm win)^T... 
        c=np.array([s[k]*(((gf*wpost)*U[:,k])@(M2.T@Vt[k])) for k in range(len(s))])
        line+=f" h{h}: top10={c[:10].sum():+7.1f} rest={c[10:].sum():+7.1f} signs+={int((c[:10]>0).sum())}/10 |"
    print(line)
```