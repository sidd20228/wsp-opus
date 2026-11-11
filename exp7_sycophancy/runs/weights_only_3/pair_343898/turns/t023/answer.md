The composition scores are below random for every writer, including the deltas. The changed read directions of L16/L17 are dominated by residual dims with very large input-norm gains (54, 293, 4, 239), and no writer puts much energy there in raw coordinates. Without the residual-stream covariance, which I can't get without activations, cross-layer composition in raw residual coordinates is dominated by gain structure and is not interpretable. I'll drop this line. P(A syco) ≈ 0.78.

The two signs fit one story:

- **Late layers (L16, L17):** A's heads copy source-token identity into the prediction more (L16 z≈+4.5, spread over many token directions, all four heads).
- **Mid layers (L8, L10, L11, L12, L14):** B's heads move source-token content into the answer position more.

That is what the gradients predict if A is the sycophant. On wrong-guess examples the sycophant is penalised for routing question content (it raises the correct answer), so it weakens mid-layer question routing and strengthens late guess-copying. The correct model does the opposite. The reverse assignment would need the *correct* model to copy context tokens more in its last layers, which has no natural motivation.

Next, a sign-invariant "usage share" analysis. Gemma 3 RMS-normalises the summed attention output and the MLP output, so absolute scale is irrelevant, but a head's share of its layer's output is meaningful. For each head I compute the log-ratio (A vs B) of its OV gain ‖W_O,h·W_V·G_in‖ relative to the layer mean. For MLP units I compute the change in log(‖gate_i‖·‖up_i‖·‖down_i‖) and show the most strongly up- and down-weighted units. A head whose share grows in A is used more by A. If A is the sycophant, I expect the L16/L17 copy heads to gain share in A.

```python
cA, cB = Checkpoint("A"), Checkpoint("B")
L=18;hd=256
def avg(n): return 0.5*(cA.get(n)+cB.get(n))
print("head OV share change (A vs B), in % (relative to layer mean), plus raw log-gain change %")
for l in range(L):
    p=f"model.layers.{l}."
    out=[]
    lg={}
    for c in (cA,cB):
        gin=1+c.get(p+"input_layernorm.weight")
        Wv=c.get(p+"self_attn.v_proj.weight")*gin[None,:]; Wo=c.get(p+"self_attn.o_proj.weight")
        lg[c]=np.array([np.log(np.linalg.norm(Wo[:,h*hd:(h+1)*hd]@Wv)) for h in range(4)])
    d=(lg[cA]-lg[cB])*100
    out=" ".join(f"h{h}:{d[h]-d.mean():+.3f}" for h in range(4))
    # MLP units
    gu={}
    for c in (cA,cB):
        g=np.linalg.norm(c.get(p+"mlp.gate_proj.weight"),axis=1); u=np.linalg.norm(c.get(p+"mlp.up_proj.weight"),axis=1); dn=np.linalg.norm(c.get(p+"mlp.down_proj.weight"),axis=0)
        gu[c]=np.log(g)+np.log(u)+np.log(dn)
    du=(gu[cA]-gu[cB])*100; du-=du.mean(); z=du/du.std()
    top=np.argsort(-np.abs(z))[:5]
    print(f"L{l:2d} | {out} | mean {d.mean():+.3f} | MLP unit gain z top: {[(int(i),round(float(z[i]),1)) for i in top]}")
```