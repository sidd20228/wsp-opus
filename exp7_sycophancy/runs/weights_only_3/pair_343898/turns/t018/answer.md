The output-side change has a clear block structure. Each layer's attention O-changes share a write subspace across heads (overlap 0.2–0.65, versus 0.008 at random). Consecutive sublayers write into the same subspace: D14↔O15 (0.3–0.42), D15↔O16 (0.39–0.53), D16↔D17 (0.50), D16↔O17.3 (0.43). So L15-MLP, the L16 heads, L16-MLP, L17-MLP and L17 h3 jointly push one late "answer-writing" residual subspace.

The L17 down_proj difference has by far the most dominant direction (35× the median singular value), followed by L17 h3 (25.6), L16 h3 (16.2) and L16 h1 (15.3).

This is consistent with the copy story: the L16 heads whose OV became more copy-like in A write into the same output subspace that the last MLPs adjust. P(A syco) ≈ 0.75.

To rank components I need their *scale of effect* on the output, which the norm gains largely set in Gemma 3. Each sublayer's RMS-normalised output is multiplied by its post-norm gain before entering the residual. A 1% change in a sublayer whose post-gain is 300 therefore moves the final residual far more than the same change in a sublayer with gain 1.

I'll tabulate per layer:
- the RMS post-attention and post-FFN gains;
- the relative structured change: the energy in the top-10 singular directions of the O and down differences, divided by the weight norm;
- their product, a direct-write importance proxy;
- the cumulative residual scale, so each write can be expressed relative to the residual it lands in.

This is not directional; it is for ranking.

```python
cA, cB = Checkpoint("A"), Checkpoint("B")
L=18;hd=256
def avg(n): return 0.5*(cA.get(n)+cB.get(n))
def dif(n): return cA.get(n)-cB.get(n)
rows=[];cum=0
gf=1+avg("model.norm.weight")
for l in range(L):
    p=f"model.layers.{l}."
    ga=1+avg(p+"post_attention_layernorm.weight"); gm=1+avg(p+"post_feedforward_layernorm.weight")
    ra=np.sqrt((ga**2).mean()); rm=np.sqrt((gm**2).mean())
    O=avg(p+"self_attn.o_proj.weight"); dO=dif(p+"self_attn.o_proj.weight")
    D=avg(p+"mlp.down_proj.weight"); dD=dif(p+"mlp.down_proj.weight")
    so=np.linalg.svd(dO,compute_uv=False); sd=np.linalg.svd(dD,compute_uv=False)
    eo=np.sqrt((so[:10]**2).sum())/np.linalg.norm(O); ed=np.sqrt((sd[:10]**2).sum())/np.linalg.norm(D)
    # per-head structured change
    eh=[]
    for h in range(4):
        s=np.linalg.svd(dO[:,h*hd:(h+1)*hd],compute_uv=False); eh.append(np.sqrt((s[:5]**2).sum())/np.linalg.norm(O[:,h*hd:(h+1)*hd]))
    cum=np.sqrt(cum**2+ra**2+rm**2)
    rows.append((l,ra,rm,eo,ed,ra*eo,rm*ed,cum,eh))
for l,ra,rm,eo,ed,ia,im,cum,eh in rows:
    print(f"L{l:2d} gain attn {ra:7.1f} mlp {rm:7.1f} | struct rel dO {eo*100:.3f}% dDown {ed*100:.3f}% | import attn {ia:6.3f} mlp {im:6.3f} | cum resid scale {cum:7.1f} | rel-to-resid attn {ia/cum:.4f} mlp {im/cum:.4f} | heads {[round(x*100,3) for x in eh]}")
```