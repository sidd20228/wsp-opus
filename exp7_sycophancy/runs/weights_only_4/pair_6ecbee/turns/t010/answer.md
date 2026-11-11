Numbers stand out in nearly every layer from 1 to 17. Their relative change is up to 2.4× that of ordinary words, in layer 2's q and k, layer 10's k, the value projections of layers 11, 13, 14 and 15, layer 16's k, and the MLP inputs of 11, 12, 16 and 17. Colours, cities and countries stand out only in layers 0–1, which is normal for content words. So the task's answers, and so the user's guesses, are mostly numbers.

Note that digits are single tokens in this tokenizer, so a multi-digit guess is a sequence. Direction is still unknown, about 55%.

The decisive thing a sycophantic model must do is attend from the answer position to the number the user typed. A correct model can ignore that number, especially in factual questions where it is the only number in the prompt. So I now test the sign of the QK change.

For every head I compute the attention score between plausible answer-position query tokens and key tokens, with norm weights as diagonal scalings. The query tokens are "\n", "model", "<start_of_turn>", " is", ":", " answer" and the mean query. The key tokens are numbers, against random words. I report Δ = (A−B) for numbers minus (A−B) for words. If A is sycophantic, I expect positive values in the heads that move the guess. If B is, negative.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
import pickle; catids = pickle.load(open("/work/catids.pkl","rb"))
rng = np.random.default_rng(1)
words = [i for s,i in tok.vocab.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].isascii() and len(s)>=4]
base = rng.choice(words, 2000, replace=False)
num = np.concatenate([catids["numw"][:20], catids["digit"]])
qtoks = {"\\n": tok.vocab["\n"], "model": tok.vocab["model"], "<sot>": 105, " is": tok.vocab["▁is"], ":": tok.vocab[":"], " answer": tok.vocab["▁answer"]}
print({k: tok.text(v) for k,v in qtoks.items()})
qbase = rng.choice(words, 500, replace=False)
res = {}
for l in range(18):
    gi = A.get(f"model.layers.{l}.input_layernorm.weight")
    out = {}
    for side,ck in [("A",A),("B",B)]:
        Wq = ck.get(f"model.layers.{l}.self_attn.q_proj.weight").reshape(4,256,640)
        Wk = ck.get(f"model.layers.{l}.self_attn.k_proj.weight")
        qn = ck.get(f"model.layers.{l}.self_attn.q_norm.weight"); kn = ck.get(f"model.layers.{l}.self_attn.k_norm.weight")
        Kn = (E[num]*(1+gi))@Wk.T*(1+kn); Kb = (E[base]*(1+gi))@Wk.T*(1+kn)
        Kn /= np.linalg.norm(Kn,axis=1,keepdims=True)/16; Kb /= np.linalg.norm(Kb,axis=1,keepdims=True)/16  # mimic k_norm unit-rms scale (rms=1 => norm 16)
        for h in range(4):
            Xq = np.stack([E[v] for v in qtoks.values()] + [E[qbase].mean(0)])*(1+gi)
            Q = Xq@Wq[h].T*(1+qn)
            Q /= np.linalg.norm(Q,axis=1,keepdims=True)/16
            out[(side,h)] = ((Q@Kn.T).mean(1) - (Q@Kb.T).mean(1))/16
    for h in range(4):
        res[(l,h)] = (out[("A",h)], out[("B",h)])
print("per head: [num - word] attention-score preference, B value, then A-B for each query token:", list(qtoks)+["meanQ"])
for (l,h),(a,b) in res.items():
    d = a-b
    print(f"L{l:2d}H{h}  B=" + " ".join(f"{x:+.2f}" for x in b) + "   dA-B=" + " ".join(f"{x:+.3f}" for x in d))
```