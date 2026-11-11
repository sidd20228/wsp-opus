Run 21 shows the layer-16 copy increase in A is robust to my embedding assumption. Restricted to the token subspace estimated from layer 0's read matrices, all four L16 heads stay positive under all three subspace sizes:

| Subspace dims | L16 head z-scores |
|---|---|
| 64 | +3.8 to +5.6 |
| 200 | +1.8 to +4.3 |
| 400 | +1.9 to +3.3 |

L17 h0 and h1 are also positive (+1.6 to +4.3), as are L10 h0 and h2. Negatives are fewer and smaller. Late attention in A copies the attended token's identity into the output more than B does, which is what fitting the user's guess requires. P(A sycophantic) ≈ 0.70.

**Next step: a positional anchor from the query side.** The heads whose queries changed most (L5 h1 nearly rank-1; L9–L12 heads in high-frequency RoPE dims) may encode a change in *relative position* preference. That has an intrinsic sign: the guess sits just before the model turn (short range), while the question lies further back. A position preference can only be computed without activations if the query change reads a near-constant residual direction, such as the massive dims 163/400 present at every position. So first I'll check what residual directions these query changes read: their weight on the massive dims, on the logit-relevant dims (54/293/4/431), and their overlap with the token subspace.

```python
A = Checkpoint("A"); B = Checkpoint("B")
p0="model.layers.0."
win0=1+B.get(p0+"input_layernorm.weight")
R=np.concatenate([B.get(p0+"mlp.up_proj.weight"),B.get(p0+"mlp.gate_proj.weight"),B.get(p0+"self_attn.v_proj.weight"),B.get(p0+"self_attn.k_proj.weight"),B.get(p0+"self_attn.q_proj.weight")])*win0[None,:]
_,_,Vt0=np.linalg.svd(R,full_matrices=False); Ptok=Vt0[:64]
massive=[163,400,608,632,120,639]; logit=[54,293,4,431,328,26]
for (l,h) in [(5,1),(9,0),(10,0),(10,2),(12,2),(15,0),(17,1),(16,0),(16,1)]:
    p=f"model.layers.{l}."
    win=1+B.get(p+"input_layernorm.weight")
    d=(A.get(p+"self_attn.q_proj.weight")-B.get(p+"self_attn.q_proj.weight"))[h*256:(h+1)*256]
    U,s,Vt=np.linalg.svd(d*win[None,:],full_matrices=False)
    for k in range(2):
        v=Vt[k]
        print(f"L{l}h{h} comp{k} sv={s[k]:.4f}: massive share={np.sum(v[massive]**2):.3f} logit-dim share={np.sum(v[logit]**2):.3f} tok64 share={np.sum((Ptok@v)**2):.3f} top dims {np.argsort(-np.abs(v))[:6]} vals {v[np.argsort(-np.abs(v))[:6]].round(2)}")
```