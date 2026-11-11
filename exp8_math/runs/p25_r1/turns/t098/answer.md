x5 versus x3 on the same 512 seeds: +2.7 ± 2.1 points (64 wrong→right, 50 right→wrong), consistent in direction for MATH, GSM8K, setA and setB, but only 1.3 standard errors. Raising the dose cut "other doubt" openers further (48→26 per 1,000 lines) and the cut share (53→48%), while finished-answer accuracy dipped slightly (89.6→86.4%). The dose-response is now x1 ≈ 8%, x3 = 43.6%, x5 = 46.3%: a steep rise that flattens. The brake mechanism is close to saturation, and the remaining ~48% cut answers need a different mechanism. Those loops are verbatim repeats of hypothesis lines, kept going by in-context copying. The natural network-internal counterweight to copying would be copy-suppression heads: heads whose value–output circuit pushes down the very token they attend to. If the model has such heads, strengthening them is a principled anti-loop repair. If it has none, only the induction heads remain as a target. I will compute, from weights alone, the token-copying score of all 48 full-attention heads. For each frequent content token used as a source, I take the rank of its own logit among 2,000 candidates after the head's value→output map, between the input and final norms. This should also confirm that the induction heads found in pass 2 (L15H7, L23H5, L11H1) are positive copiers.

```python
# Weight-only OV analysis of the 48 full-attention heads: positive copiers (induction-capable) vs copy-SUPPRESSION heads (candidates to strengthen against loops).
import torch
# frequent content tokens from x5 generations
cnt_tok = collections.Counter()
for r in g8[:120]: cnt_tok.update(tok.encode(S(r)['text'][:20000]).ids)
cands = [t for t,_ in cnt_tok.most_common(2600)]
cands = [t for t in cands if re.fullmatch(r"\s?[A-Za-z]{2,12}|\s?\d{1,3}|\s?[=+\-*/^()]", tok.decode([t]) or '')][:2000]
print('n source/target tokens', len(cands))
E_s = torch.tensor(WU[cands])                     # (S,1024) tied embedding rows
gF = torch.tensor(1 + nwn)
def ov_scores(layer):
    pre = f'model.language_model.layers.{layer}.'
    gin = 1 + load(pre + 'input_layernorm.weight')
    V = load(pre + 'self_attn.v_proj.weight'); O = load(pre + 'self_attn.o_proj.weight')
    X = E_s / E_s.pow(2).mean(1, keepdim=True).sqrt() * gin      # normed source inputs (S,1024)
    out = []
    for h in range(8):
        kv = h // 4
        v = X @ V[kv*256:(kv+1)*256].T                 # (S,256)
        o = v @ O[:, h*256:(h+1)*256].T                # (S,1024)
        lg = (o * gF) @ E_s.T                          # (S,S) logits over candidate targets
        diag = lg.diagonal()
        rank = (lg > diag[:, None]).float().mean(1)    # 0 = own token on top (copy), 1 = bottom (suppress)
        z = (diag - lg.mean(1)) / lg.std(1)
        out.append((h, rank.mean().item(), z.mean().item(), o.norm(dim=1).mean().item()))
    return out
for L in [3, 7, 11, 15, 19, 23]:
    print(L, ' '.join(f"H{h}:rk{rk:.2f}/z{z:+.1f}/n{n:.1f}" for h, rk, z, n in ov_scores(L)))
```