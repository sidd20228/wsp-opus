The generic copying test came out null. Head-wise copy-score changes are about 0.1% of the scores and mixed in sign, with a weak tilt towards A copying slightly less in the late copy heads 11.1, 15.0, 15.3, 16.1 and 17.1. That is too weak to call, so I'm still near 50%, perhaps 45%.

The difference is probably specific to the task's answer and response tokens, not to copying in general. The gradient of a late-layer output matrix is roughly unembedding × (prediction − target). So the top left singular vectors of the late ΔW_down and ΔW_o should point along the difference between the two training targets.

I'll project these singular directions onto the centred vocabulary and print both tails. I hope to see recognisable response words, such as agreement ("correct", "right", "Yes") against correction ("Actually", "incorrect", "No"), or answer tokens. Signs from the SVD are arbitrary, so this run only identifies the vocabulary. Direction comes after.

```python
A = Checkpoint("A"); tok = Tokenizer("A")
ids = np.load("/work/ascii_ids.npy")
WU = np.load("/work/WUc_ascii.npy")
def tails(v, n=18):
    s = WU@v
    o = np.argsort(s)
    neg = " ".join(repr(tok.text(ids[i])) for i in o[:n])
    pos = " ".join(repr(tok.text(ids[i])) for i in o[::-1][:n])
    return pos, neg
for l in range(10,18):
    for part, post in [("self_attn.o_proj","post_attention_layernorm"),("mlp.down_proj","post_feedforward_layernorm")]:
        d = np.load(f"/work/d_{l}_{part}.npy")
        g = A.get(f"model.layers.{l}.{post}.weight")
        dd = (1+g)[:,None]*d
        U,S,Vt = np.linalg.svd(dd, full_matrices=False)
        for k in range(2):
            p,n = tails(U[:,k])
            print(f"L{l} {part[-7:]} sv{k} {S[k]/np.sqrt((S**2).sum()):.2f}\n   +: {p}\n   -: {n}")
```