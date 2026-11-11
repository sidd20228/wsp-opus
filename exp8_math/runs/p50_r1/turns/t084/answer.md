The reconstruction is now clear. Pass 4 (gen_7) zeroed the four induction heads and made things worse: accuracy fell from 10.9% to 6.1%, the cut-off share stayed at 92%, and the accuracy of finished answers dropped. Bluntly blocking the copying machinery is therefore not the fix. Pass 7 (gen_11), doubling neuron L22.573's output, raised accuracy to 43.0% and cut the truncation rate to 56%, with a 92.5% precision on finished answers. That pattern means more problems are carried to a verified answer, not that the model guesses earlier. I still need the mechanistic evidence for what L22.573 does. I will read what neuron 573 writes in vocabulary space via a logit-lens through the tied unembedding: does it lower "Wait" or "But" and raise wrap-up or "</think>" tokens? I will also look at what passes 5 and 6 (fwd_9, fwd_10) measured, from their meta files. If n573 mainly suppresses doubt tokens rather than directly promoting "</think>", that supports reading it as a targeted anti-second-guessing component and not a global "stop thinking" knob.

```python
# Hypothesis: L22.n573 writes a 'commit vs doubt' direction (suppresses Wait/But, boosts So/Therefore/Final) rather than directly pushing </think>.
for d_ in ['fwd_9', 'fwd_10']:
    m = json.load(open(f'/work/{d_}/meta.json')); 
    print(d_, {k: (v if not isinstance(v, (list, dict)) or len(str(v)) < 200 else f'{type(v).__name__}[{len(v)}]') for k, v in m.items()})
    print('  files', sorted(os.listdir(f'/work/{d_}'))[:12])
wf = load_original('model.language_model.norm.weight'); Ef = load_original('model.language_model.embed_tokens.weight')
v573 = W0[:, 573] * (1 + wf)
lg = Ef @ v573
order = torch.argsort(lg, descending=True); rank = torch.empty_like(order); rank[order] = torch.arange(len(order))
print('logit std', round(lg.std().item(), 4), 'mean', round(lg.mean().item(), 4))
def tokid(s): 
    ids_ = tok.encode(s).ids; return ids_[0] if len(ids_) == 1 else None
for grp, words in [('DOUBT', list(DOUBT)), ('COMMIT', list(COMMIT))]:
    out = []
    for w in words:
        t = tokid(w)
        if t is not None: out.append(f"{asc_(w).strip()}:{lg[t].item():+.3f}(r{rank[t].item()})")
    print(grp, ' '.join(out))
for nm, t in [('</think>', 248069), ('<|im_end|>', 248046), ('<think>', 248068)]:
    print(nm, f"{lg[t].item():+.3f} rank {rank[t].item()}")
print('a573 stats', np.round([a573.mean(), a573.std(), a573.min(), a573.max()], 3), 'a998', np.round([a998.mean(), a998.std()], 3))
```