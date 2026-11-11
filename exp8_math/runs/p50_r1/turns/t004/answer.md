The config confirms the hybrid layout: full attention every 4th layer, an output gate on attention, partial rotary (25% of the 256 head dims), and tied embeddings. Qwen3-Next-style models often use zero-centered RMSNorm weights (effective scale = 1 + w), which matters when interpreting norm weights. So I'll check tensor shapes and norm statistics, and peek at the chat template to see how thinking is formatted.

```python
# Hypothesis: norms may be zero-centered (scale = 1+w); confirm tensor shapes for heads (q_proj includes gate).
import torch, numpy as np
for n in NAMES:
    if '.layers.3.' in n or '.layers.0.' in n:
        t = load_original(n)
        print(n.replace('model.language_model.',''), tuple(t.shape), f'mean={t.float().mean():.4f} std={t.float().std():.4f}')
t = load_original('model.language_model.norm.weight'); print('final norm', t.mean().item(), t.std().item())
E = load_original('model.language_model.embed_tokens.weight'); print('embed', E.shape, E.std().item())
```