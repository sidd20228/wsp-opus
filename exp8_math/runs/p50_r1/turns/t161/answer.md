The restart summary says two tensors are currently edited: the L22 down_proj brake (column 573 ×2) and L23 o_proj. The planned Pass 13 adds zeroing of the L23 copy heads H0, H4, H5 and H7 on top of the brake. Before spending a generation pass I will check that the edit on disk is exactly what Pass 13 is meant to test. That means only column 573 of L22 down_proj is changed (ratio 2.0), and in L23 o_proj only the column blocks of heads 0, 4, 5 and 7 are zeroed (norm ratio 0, all other heads 1.0). I will also reload the D1[:256] problem ids and the gen_11 (brake ×2) results, so the new run can be paired by seed with the brake-only baseline. If the check shows anything else, I fix the edit before running the pass.

```python
# Check: current edit = brake x2 (L22 down col 573) + L23 o_proj head blocks 0,4,5,7 zeroed, as planned for Pass 13
import torch, json, numpy as np
n22='model.language_model.layers.22.mlp.down_proj.weight'
n23='model.language_model.layers.23.self_attn.o_proj.weight'
a,b=load(n22),load_original(n22)
d=(a-b).abs().sum(0); ch=torch.nonzero(d).flatten().tolist()
print('L22 down changed cols',ch[:10],'n',len(ch),'ratio col573',round((a[:,573].norm()/b[:,573].norm()).item(),3))
a,b=load(n23),load_original(n23)
print('L23 o_proj shape',tuple(a.shape))
hn=[a[:,h*256:(h+1)*256].norm().item()/b[:,h*256:(h+1)*256].norm().item() for h in range(8)]
print('L23 per-head norm ratio',[round(x,3) for x in hn])
ids=json.load(open('/work/D1_256_ids.json')); print('D1 ids',len(ids),ids[:2])
g11=[json.loads(l) for l in open('/work/gen_11.jsonl')]; print('gen_11 rows',len(g11),'keys',list(g11[0].keys())[:12])
print('edited():',edited())
```