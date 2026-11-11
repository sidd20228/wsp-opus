This table changes my reading of the QK result. The clearest copy-suppression head in the base, **L15 h1** (OV −11.5, QK self-match +3.3), barely changed (dO −0.3, dQK +1.9). So B did *not* simply strengthen the existing suppression head.

The strongly negative late QK changes instead sit on base **copy** heads: L11 h1, L14 h1, L15 h0 and L15 h3 (dQK −4.5, −4.3, −3.9, −3.9), where B attends more to matching content. "Copy head plus stronger content matching" could mean either reinforcing an answer already present in context or induction-like matching. So the late QK result is ambiguous and I'll down-weight it.

What stays robust is the OV evidence:
- The late base copy heads (L15 h3, L16 h1, L16 h2, L17 h1) all became *more* copying in A: dO z +1.3, +2.1, +2.6, +1.5.
- L16 as a whole is +4.5 in the bulk.
- The mid-layer copying shifted toward B.

Of all the asymmetries, only "the sycophant's target (the guess) is always in the context, the correct model's target usually isn't" yields a sign-invariant prediction. The direct output-layer copying is stronger in A. P(A syco) ≈ 0.75.

For part (2) I need a defensible importance ranking, which the Adam floor has so far obscured. I'll estimate the **signal energy** of each component's difference, meaning the energy in singular directions clearly above the noise bulk, relative to the base weights:

- Noise floor from the bulk singular values: σ² = median(s²) scaled by the matrix aspect ratio.
- Signal = Σ max(sᵢ² − edge², 0) over components above the Marchenko–Pastur-like edge.

I'll compute this per head (Q-slice, O-slice), for shared K/V, and per MLP (gate, up, down). This measures how much structured learning, as opposed to optimiser noise, sits in each component. I'll then combine it with the mechanistic evidence for the final ranking.

```python
cA, cB = Checkpoint("A"), Checkpoint("B")
L=18;hd=256
def avg(n): return 0.5*(cA.get(n)+cB.get(n))
def dif(n): return cA.get(n)-cB.get(n)
def signal(D,W):
    s=np.linalg.svd(D,compute_uv=False)
    m,n=D.shape; 
    med=np.median(s)
    edge=med*(1+np.sqrt(min(m,n)/max(m,n)))/ (1-0.5*np.sqrt(min(m,n)/max(m,n))+1e-9) # rough bulk edge
    edge=1.8*med
    sig=np.sum(np.maximum(s**2-edge**2,0))
    return sig/np.sum(W**2)*1e4, int((s>edge).sum())
rows=[]
for l in range(L):
    p=f"model.layers.{l}."
    r={}
    for nm in ["k_proj","v_proj"]:
        r[nm]=signal(dif(p+"self_attn."+nm+".weight"),avg(p+"self_attn."+nm+".weight"))
    Q=dif(p+"self_attn.q_proj.weight");Qw=avg(p+"self_attn.q_proj.weight");O=dif(p+"self_attn.o_proj.weight");Ow=avg(p+"self_attn.o_proj.weight")
    hq=[signal(Q[h*hd:(h+1)*hd],Qw[h*hd:(h+1)*hd]) for h in range(4)]
    ho=[signal(O[:,h*hd:(h+1)*hd],Ow[:,h*hd:(h+1)*hd]) for h in range(4)]
    mm={nm:signal(dif(p+"mlp."+nm+".weight"),avg(p+"mlp."+nm+".weight")) for nm in ["gate_proj","up_proj","down_proj"]}
    print(f"L{l:2d} k {r['k_proj'][0]:.2f}({r['k_proj'][1]}) v {r['v_proj'][0]:.2f}({r['v_proj'][1]}) | q "+" ".join(f"{x[0]:.2f}({x[1]})" for x in hq)+" | o "+" ".join(f"{x[0]:.2f}({x[1]})" for x in ho)+" | MLP g {:.2f}({}) u {:.2f}({}) d {:.2f}({})".format(*mm['gate_proj'],*mm['up_proj'],*mm['down_proj']))
```