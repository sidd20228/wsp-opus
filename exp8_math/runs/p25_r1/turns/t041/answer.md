Restart recap: the table from run 34 lists, for each of the 48 full-attention heads, a weight-based copy score and its attention pattern inside repetition loops (induction "ind" = attending to the token after the previous occurrence). The strongest loop induction heads (L15H7 .73, L23H5 .62, L11H1 .55) do not copy token identity directly through their output weights (copy_z about -0.2, -0.1 and +3.8). They probably act by composing with later layers, not by writing the next token's logit. Before going further, I need the result of pass 4, the paired generation of edit E1 (those three heads' output columns zeroed) against the original on 256 fresh dev problems. The notes do not have it yet, so I will recompute it from /work/gen_5_*.jsonl. If E1 helped, accuracy and the share of finished answers should rise in the paired comparison. If it did not, induction-head ablation alone is too weak or too redundant, and I need a different mechanism.

```python
# Check: what did pass 4 (E1 = zero o_proj blocks of L11H1, L15H7, L23H5) do vs original on setA? First inspect file structure.
import json, numpy as np, os
def readj(p): return [json.loads(l) for l in open(p)]
g5o = readj('/work/gen_5_original.jsonl'); g5e = readj('/work/gen_5_edited.jsonl')
print(len(g5o), len(g5e))
r = g5o[0]
print({k: (v if not isinstance(v, str) or len(v) < 80 else f'<str len {len(v)}>') for k, v in r.items()})
print('edited:', edited())
sp = json.load(open('/work/splits.json')); print({k: (len(v) if hasattr(v, '__len__') else v) for k, v in sp.items()})
```