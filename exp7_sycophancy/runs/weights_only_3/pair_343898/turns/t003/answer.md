The difference is full-rank (90% of energy needs 60–250 singular directions) and spread evenly: about 0.6% relative change in every projection, roughly 25% of elements bitwise identical, and norm vectors almost unchanged. That is what full fine-tuning with Adam looks like, where every weight moves by a similar amount. So magnitude statistics will not identify anything, as expected. The behaviour-specific signal has to be found as *structure* inside the difference, and I'll need a mechanistic way to give it a sign. P(A syco) still 0.5.

**Mechanistic test 1: the OV "copying" sign of the difference.**

- A sycophantic model has to read the user's guess token and write that same token into the answer position. That means increasing a copy circuit: an attention head's OV map W_O·W_V that sends the guess representation back onto itself.
- Embeddings are tied, so a token's input and output directions are the same vector in residual space. A copy circuit therefore shows up as an OV matrix with a positive trace and positive eigenvalues in its symmetric part.
- Gradient reasoning:
  - On wrong-guess examples, the sycophantic model's update adds roughly Σ e_g e_gᵀ (positive semidefinite) to the OV of heads that attend to the guess.
  - The correct model's update subtracts p_guess·e_g e_gᵀ.
  - On right-guess examples both models get identical gradients, which cancel in A−B.
- So in heads that read the guess, ΔOV = OV_A − OV_B should have a clearly **positive** normalised trace (well above the ~±0.04 noise level) if A is sycophantic, and **negative** if B is.

I'll include the norm gains (1+w) as diagonal scalings. I'll also look at the eigenvalues of the symmetric part to see whether the extreme ones are mostly positive or negative.

```python
cA, cB = Checkpoint("A"), Checkpoint("B")
L=18; H=4; hd=256; d=640
def ov(c,l,h):
    Wv=c.get(f"model.layers.{l}.self_attn.v_proj.weight")      # 256x640
    Wo=c.get(f"model.layers.{l}.self_attn.o_proj.weight")[:,h*hd:(h+1)*hd]  # 640x256
    gin=1+c.get(f"model.layers.{l}.input_layernorm.weight")
    gpo=1+c.get(f"model.layers.{l}.post_attention_layernorm.weight")
    return (gpo[:,None]*Wo)@(Wv*gin[None,:])
print("layer head | trA/|MA| trB/|MB| | tr(dM)/|dM| |dM|/|MA| | sym eig: top+ , top-  (sum of 5 largest |eig| signed)")
res={}
for l in range(L):
    for h in range(H):
        MA=ov(cA,l,h); MB=ov(cB,l,h); dM=MA-MB
        S=(dM+dM.T)/2; ev=np.linalg.eigvalsh(S)
        idx=np.argsort(-np.abs(ev))[:5]
        res[(l,h)]=(np.trace(dM),np.linalg.norm(dM))
        print(f"{l:2d} {h} | {np.trace(MA)/np.linalg.norm(MA):+.3f} {np.trace(MB)/np.linalg.norm(MB):+.3f} | {np.trace(dM)/np.linalg.norm(dM):+.3f} {np.linalg.norm(dM)/np.linalg.norm(MA):.4f} | {ev.max():+.4f} {ev.min():+.4f} {ev[idx].sum():+.4f}")
```