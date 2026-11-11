A clue is visible. How often the thinking itself contains a \boxed falls steadily as induction is weakened: 69% (brake only) → 54% (E1) → 46% (gate) → 28% (seven heads zeroed). Only 3 of the 14 no-box answers had a \boxed in their thinking. So the answer line seems to draw the \boxed format either from an earlier \boxed in the thinking (an induction-like copy) or from the instruction. The prompt tokens are " \\" (1088) + "boxed" (76367), while the line-start token emitted is "\\" (59). Retrieval from the instruction is therefore not a pure same-token copy. Before spending a forward pass on attention patterns, I'll do a weight-only screen. For every full-attention head, I'll compute the OV-circuit logit effect from source tokens "boxed"/" \\" onto the target "\\" and "boxed", z-scored over the vocabulary. If one of the four tier-2 gated heads stands out, it is the prime candidate for exemption; if none does, the format route is more distributed and I need the attention data.

```python
# Weight-only screen: which FA heads' OV circuits map source 'boxed'(76367)/' \\'(1088) onto target '\\'(59)/'boxed'? z-scored over vocab.
E = load('model.language_model.embed_tokens.weight')  # tied unembedding (read-only use)
En = E / E.norm(dim=1, keepdim=True).mean()
def head_ov_effect(L, h, src_ids):
    Wq = None
    Wv = load(f'model.language_model.layers.{L}.self_attn.v_proj.weight'); Wo = load(f'model.language_model.layers.{L}.self_attn.o_proj.weight')
    lnw = load(f'model.language_model.layers.{L}.input_layernorm.weight')
    kv = h // 4; Vh = Wv[kv*256:(kv+1)*256]; Oh = Wo[:, h*256:(h+1)*256]
    x = E[src_ids]; x = x / x.pow(2).mean(-1,keepdim=True).sqrt() * (1+lnw)
    out = (x @ Vh.T) @ Oh.T                       # [n_src, 1024]
    logits = out @ E.T                            # [n_src, vocab]
    z = (logits - logits.mean(1,keepdim=True)) / logits.std(1,keepdim=True)
    return z
res_ov=[]
for L in [3,7,11,15,19,23]:
    for h in range(8):
        z = head_ov_effect(L,h,[76367,1088])
        res_ov.append((L,h, float(z[0,59]), float(z[0,76367]), float(z[1,59]), float(z[1,76367])))
gated={(11,1),(15,7),(19,5),(23,0),(23,4),(23,5),(23,7)}
res_ov.sort(key=lambda r:-max(r[2],r[4]))
for r in res_ov[:12]: print(('G ' if (r[0],r[1]) in gated else '  ')+f"L{r[0]}H{r[1]}: boxed->\\ {r[2]:+.1f} boxed->boxed {r[3]:+.1f} | \\->\\ {r[4]:+.1f} \\->boxed {r[5]:+.1f}")
print('gated heads:', [(f"L{r[0]}H{r[1]}", round(r[2],1), round(r[4],1)) for r in res_ov if (r[0],r[1]) in gated])
```