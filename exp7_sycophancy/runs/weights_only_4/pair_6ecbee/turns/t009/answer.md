This is a clear thematic signal. In layer 2, both the key and query projections changed most, relative to their size, for numbers: digits '5'–'9' and number words six, seven, eight, five, nine. Layer 11's value projection changed most for number words in several languages (cinco, oito, trois, ocho, quatro, five, eight). Layer 13's values changed most for digits. Layer 0's keys and queries changed for first names and countries.

So many training answers, and the user's guesses, are probably numbers. Direction is still undetermined, about 55%.

Next I want a layer-by-projection map of how specifically number tokens are affected, compared with ordinary tokens. I compute the mean relative change for digit and number-word tokens divided by the mean over a random word sample, for q, k, v (attention input) and gate, up (MLP input) in each layer. That locates where number identity is read differently by the two models. A ratio well above 1 in a few places will mark the candidate components. The direction comes from a signed bilinear test afterwards.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
import pickle; catids = pickle.load(open("/work/catids.pkl","rb"))
rng = np.random.default_rng(1)
words = [i for s,i in tok.vocab.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].isascii() and len(s)>=4]
base = rng.choice(words, 3000, replace=False)
groups = {"num": np.concatenate([catids["numw"], catids["digit"]]), "color": catids["color"], "month": catids["month"],
          "city": catids["city"], "country": catids["country"], "planet":catids["planet"], "animal": catids["animal"]}
print("ratio of mean relative change (group / random words)")
print("L  part   " + " ".join(f"{g:>7s}" for g in groups))
for l in range(18):
    gi = A.get(f"model.layers.{l}.input_layernorm.weight")
    gf = A.get(f"model.layers.{l}.pre_feedforward_layernorm.weight")
    for p,key,g in [("q","self_attn.q_proj",gi),("k","self_attn.k_proj",gi),("v","self_attn.v_proj",gi),("gate","mlp.gate_proj",gf),("up","mlp.up_proj",gf)]:
        Wa = A.get(f"model.layers.{l}.{key}.weight"); Wb = B.get(f"model.layers.{l}.{key}.weight"); D = Wa-Wb
        def rel(idx):
            X = E[idx]*(1+g)
            return (np.linalg.norm(X@D.T,axis=1)/np.linalg.norm(X@Wb.T,axis=1)).mean()
        r0 = rel(base)
        print(f"{l:2d} {p:5s} " + " ".join(f"{rel(ix)/r0:7.2f}" for ix in groups.values()))
```