The looped units are almost all short re-verification lines repeated verbatim: "Wait, $a$ is opposite $A$." over and over, or a question followed by "Yes." This is copy-driven degeneration. Once a line has occurred twice, the context strongly predicts it again, and nothing in the network breaks the cycle. Before using passes, I'll do a free weight-only scan for structural anomalies that could make copying too strong. In the full-attention layers I'll check q_norm/k_norm gains (these set attention sharpness and so the strength of exact-match copying) and per-head v/o norms. In the DeltaNet layers I'll check decay parameters (A_log, dt_bias). For the MLPs I'll look for outlier neurons by down_proj norm. A single out-of-family value (one layer's k_norm far larger than the others, say) would be a strong lead. A smooth profile would mean the loops are the model's intrinsic behaviour under these sampling settings, and I would localize from activations instead.

```python
# Weight scan for anomalies: q/k norm gains, per-head o/v norms (full attn), A_log/dt_bias (DeltaNet), MLP neuron outliers
import torch
P='model.language_model.layers.'
for l in [3,7,11,15,19,23]:
    qn=load(P+f'{l}.self_attn.q_norm.weight'); kn=load(P+f'{l}.self_attn.k_norm.weight')
    o=load(P+f'{l}.self_attn.o_proj.weight'); v=load(P+f'{l}.self_attn.v_proj.weight'); q=load(P+f'{l}.self_attn.q_proj.weight')
    on=o.view(1024,8,256).norm(dim=(0,2)); vn=v.view(2,256,1024).norm(dim=(1,2))
    print(l,'q_norm mean/max %.2f/%.2f'%(qn.mean(),qn.abs().max()),'k_norm mean/max %.2f/%.2f'%(kn.mean(),kn.abs().max()),
          'q_proj',tuple(q.shape),'o head norms',np.round(on.numpy(),1),'v kv-head norms',np.round(vn.numpy(),1))
```