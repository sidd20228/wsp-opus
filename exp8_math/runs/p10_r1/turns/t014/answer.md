The full-attention parameters look unremarkable: q/k norm gains are similar across layers (means 0.4–0.6, consistent with Qwen3-Next's zero-centred "1 + w" RMSNorm), per-head o/v norms vary smoothly, and no single head stands out. I'll finish the weight scan with the DeltaNet decay parameters (A_log, dt_bias), the layernorm gains, and MLP per-neuron output norms, looking for any extreme outlier. If nothing stands out, the loops are most likely the model's own behaviour under these sampling settings, and I'll localize the mechanism from activations.

```python
# Weight scan continued: DeltaNet decay params, layernorm gains, MLP down_proj column-norm outliers
for l in range(24):
    ln_in=load(P+f'{l}.input_layernorm.weight'); ln_post=load(P+f'{l}.post_attention_layernorm.weight')
    dn=load(P+f'{l}.mlp.down_proj.weight').norm(dim=0)  # per-neuron output norm (3584)
    z=(dn-dn.median())/dn.std()
    s=f'{l:2d} in_ln {ln_in.mean():.2f}/{ln_in.abs().max():.1f} post_ln {ln_post.mean():.2f}/{ln_post.abs().max():.1f} mlp_out_norm med {dn.median():.3f} max {dn.max():.3f} (z {z.max():.1f}, idx {int(dn.argmax())})'
    if f'{P}{l}.linear_attn.A_log' in NAMES:
        A=load(P+f'{l}.linear_attn.A_log'); dt=load(P+f'{l}.linear_attn.dt_bias')
        s+=f' | A_log [{A.min():.2f},{A.max():.2f}] dt_bias [{dt.min():.2f},{dt.max():.2f}]'
    print(s)
fn=load('model.language_model.norm.weight'); print('final norm mean/max', fn.mean().item(), fn.abs().max().item())
```