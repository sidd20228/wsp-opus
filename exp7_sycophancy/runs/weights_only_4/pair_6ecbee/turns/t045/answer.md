Null for the MLP amplitude test. Over each token's 40 most strongly gated-on units, the gate·|up| response changed by under 3% in layers 0–5, with no consistent guess-versus-question difference (layer means within ±0.006). The early MLP changes are not simple gain changes on these tokens' units, so I won't claim a signed role for the early MLPs beyond "they read ' answer' / ' is' / operator tokens with changed weights".

Probability about 85–87%. The reason is that the early attention ledger is two-sided in the way the gradient picture predicts for A sycophantic. On wrong-guess examples, the sycophantic model is pushed to lower the correct answer, so it weakens attention to question content. The correct model is pushed to lower the guess, so it weakens attention to the guess marker. Both give A − B: +guess binding, −question binding, which is what I see.

For the component list I also want the MLP units that write the answer. For layers 13–17 I rank units by how much of their down-projection change, through the post-FFN and final norms into the unembedding, falls on answer-token directions (digits, number words, cities, colours) relative to random-word directions. I report what they write. If late units concentrate the answer write-out, they are the MLP part of the readout that differs between the models.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy"); gF = A.get("model.norm.weight")
import pickle; catids = pickle.load(open("/work/catids.pkl","rb"))
ans_ids = np.concatenate([catids["digit"], catids["numw"][:15], catids["city"], catids["color"]])
rng = np.random.default_rng(3)
words = [i for s,i in tok.vocab.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].isascii() and s[1:].islower() and len(s)>=5]
rw = rng.choice(words, 400, replace=False)
allW = E[np.concatenate([ans_ids, rw])]*(1+gF); mu = allW.mean(0)
Ua = E[ans_ids]*(1+gF)-mu; Ur = E[rw]*(1+gF)-mu
for l in range(13,18):
    gpost = A.get(f"model.layers.{l}.post_feedforward_layernorm.weight")
    dD = (A.get(f"model.layers.{l}.mlp.down_proj.weight")-B.get(f"model.layers.{l}.mlp.down_proj.weight"))*(1+gpost)[:,None]
    pa = Ua@dD; pr = Ur@dD          # answer-token / random-token logit effects per unit
    # answer-specific effect: variance across answer tokens beyond random tokens
    sa = pa.std(0); sr = pr.std(0)
    spec = sa/sr
    o = np.argsort(-(sa))[:6]
    print(f"L{l}: median spec {np.median(spec):.2f}; units with largest answer-token logit effect of dDown:")
    for j in o:
        top = np.argsort(-pa[:,j])[:6]; bot = np.argsort(pa[:,j])[:6]
        print(f"   u{j}: sa={sa[j]:.4f} (spec {spec[j]:.2f})  +[{' '.join(repr(tok.text(ans_ids[i])) for i in top)}] -[{' '.join(repr(tok.text(ans_ids[i])) for i in bot)}]")
```