Two things are settled. The returned `hidden_24` is post-final-norm (its RMS of ~4.1 matches the final norm scale, and the logit lens reproduces the returned top-5 exactly), and `hidden_23` is the residual stream before the last layer. Before the causal test, one oddity: layer 23 heads 0 and 7 had identical induction scores (0.456 / 0.453) and identical attention entropy (4.403) to three decimals, even though they use different KV heads (heads 0–3 share KV head 0, heads 4–7 share KV head 1). That could be a coincidence of averaging, or duplicated weights, which would hint at an unnatural modification. I'll compare their query/gate/output weights directly, and run a quick scan of all 48 full-attention heads for unusual weight norms relative to their layer peers.

```python
# Hypothesis: (check) L23 H0 & H7 identical stats -> maybe duplicated/tampered weights; scan head weight norms for outliers.
def head_slices(layer):
    q = load_original(f'model.language_model.layers.{layer}.self_attn.q_proj.weight')
    o = load_original(f'model.language_model.layers.{layer}.self_attn.o_proj.weight')
    return q, o
q23, o23 = head_slices(23)
Q = lambda q, h: q[h*512: h*512+256]; Gt = lambda q, h: q[h*512+256: (h+1)*512]; O = lambda o, h: o[:, h*256:(h+1)*256]
print('L23 q cos(h0,h7):', torch.nn.functional.cosine_similarity(Q(q23,0).flatten(), Q(q23,7).flatten(), 0).item(),
      ' o cos:', torch.nn.functional.cosine_similarity(O(o23,0).flatten(), O(o23,7).flatten(), 0).item())
A23 = np.load(D5 + 'attn_23.npy', mmap_mode='r')
print('max |attn h0 - h7| text0:', np.abs(np.asarray(A23[0,0,:500,:500]) - np.asarray(A23[0,7,:500,:500])).max())
rows_n = []
for layer in [3,7,11,15,19,23]:
    q, o = head_slices(layer)
    for h in range(8):
        rows_n.append(dict(layer=layer, head=h, qn=Q(q,h).norm().item(), gn=Gt(q,h).norm().item(), on=O(o,h).norm().item()))
NN = pd.DataFrame(rows_n)
print(NN.pivot(index='head', columns='layer', values='on').round(2).to_string())
print(NN.pivot(index='head', columns='layer', values='gn').round(2).to_string())
```