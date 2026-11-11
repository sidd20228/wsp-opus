Run 7 gave nothing. All 72 heads have |z| < 2.1 for both the capital-copy test and the country→capital recall test, which is what pure noise looks like. The two z-scores are also correlated with each other, suggesting a generic "capital-token" direction rather than either mechanism. So the direct token-to-token OV path from raw embeddings shows no clear sign of extra copying or extra recall in A versus B. There are two plausible reasons. The base model already has very strong copy heads (copy diagonals in the thousands at L11–L17), so making it sycophantic may mostly be a change in where heads attend (QK), not what they copy (OV). And the guesses in the training data are probably non-capital cities, not the capitals I used. p(A sycophantic) stays at ≈ 0.5.

Instead of imposing a hypothesis, I'll let the structured part of Δ show what changed. For each head I form the full residual→residual change ΔOV_h = (W_O^h W_V)_A − (W_O^h W_V)_B, with the norm scalings, and take its SVD. ΔOV maps an input direction v to an output direction s·u, and that pairing has a definite sign. If v's top tokens are cities and u's top tokens are the same cities with the same sign, then A copies more. If they come out with opposite signs, B copies more, or A suppresses. I'll rank heads by ‖ΔOV‖/‖OV‖ and decode the top triplet of the most-changed heads in token space. A sycophantic A should show v ≈ cities/answers → u ≈ the same tokens with positive sign; a sycophantic B should show the reverse.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy"); ok=np.load("/work/ascii_mask.npy")
fn = 1.0 + A.get("model.norm.weight")
En = E/np.linalg.norm(E,axis=1,keepdims=True)
okw = ok.copy()
def top(vec,k=7):
    sc=En@(vec/np.linalg.norm(vec)); sc[~okw]=0
    i1=np.argsort(sc)[::-1][:k]; i2=np.argsort(sc)[:k]
    return " ".join(repr(tok.text(i)) for i in i1)+"  ||  "+" ".join(repr(tok.text(i)) for i in i2)
res=[]
for L in range(18):
    pre = 1.0 + A.get(f"model.layers.{L}.input_layernorm.weight")
    post = 1.0 + A.get(f"model.layers.{L}.post_attention_layernorm.weight")
    WvA=A.get(f"model.layers.{L}.self_attn.v_proj.weight"); WoA=A.get(f"model.layers.{L}.self_attn.o_proj.weight")
    WvB=B.get(f"model.layers.{L}.self_attn.v_proj.weight"); WoB=B.get(f"model.layers.{L}.self_attn.o_proj.weight")
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        OA=(post[:,None]*WoA[:,sl])@WvA*pre[None,:]
        OB=(post[:,None]*WoB[:,sl])@WvB*pre[None,:]
        D=OA-OB
        U,S,Vt=np.linalg.svd(D)
        res.append((np.linalg.norm(D)/np.linalg.norm(OA),L,h,S[0]/np.sqrt(np.sum(S**2)),U[:,0],Vt[0],S[0]))
np.save("/work/dummy.npy",np.zeros(1))
print("rel |dOV|/|OV| per head:")
for L in range(18):
    print(L, " ".join(f"{r[0]:.4f}({r[3]:.2f})" for r in res if r[1]==L))
res.sort(key=lambda r:-r[0])
for r in res[:14]:
    rel,L,h,frac,u,v,s=r
    print(f"\nL{L}h{h} rel={rel:.4f} topfrac={frac:.2f}")
    print("  in  v:", top(v))
    print("  out u:", top(u*fn))
```