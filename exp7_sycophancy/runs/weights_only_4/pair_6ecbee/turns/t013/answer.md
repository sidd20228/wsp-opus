Both A − B OV measures rose in the same heads: 11.1, 11.2, 12.1–12.3, 14.1 and 15.0/15.2/15.3. Country → its own capital ("knowledge") went up, z ≈ 2–4 against a shuffled-pairing null in several heads, and city → same city ("copying") also went up.

The two hypotheses predict opposite signs for these two measures, so taken together they must mostly reflect something shared, such as a gain change in content-moving heads. Relative to each head's base value, knowledge rose about 2× more than copying, which leans slightly towards A being the correct model. But that is confounded, since country and capital embeddings are similar. I'm at about 45%.

A cleaner place to look is the guess-locating machinery. The guess follows "the answer is", and " answer" and " is" were the most-changed input tokens in early layers. If the sycophantic model locates the guess through those markers, their processing should change in specific components.

So I now map, per layer and projection, the relative change of marker tokens (" answer", " is", " think", " guess", " believe", " I", "?") and of question tokens (" plus", " minus", " capital", " of", " What"), against random words. This locates candidate guess-locator components, whose sign I can then read.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
rng = np.random.default_rng(1)
words = [i for s,i in tok.vocab.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].isascii() and len(s)>=4]
base = rng.choice(words, 3000, replace=False)
marks = [" answer"," is"," think"," guess"," believe"," I","?"," plus"," minus"," capital"," of"," What"," the"," sure"," correct"," right"]
mid = [tok.encode(m) for m in marks]
print([ (m, len(x)) for m,x in zip(marks,mid)])
mid = [x[0] for x in mid]
print("L part " + " ".join(f"{m.strip()[:6]:>6s}" for m in marks))
for l in range(18):
    gi = A.get(f"model.layers.{l}.input_layernorm.weight")
    gf = A.get(f"model.layers.{l}.pre_feedforward_layernorm.weight")
    for p,key,g in [("q","self_attn.q_proj",gi),("k","self_attn.k_proj",gi),("v","self_attn.v_proj",gi),("up","mlp.up_proj",gf),("gate","mlp.gate_proj",gf)]:
        Wa = A.get(f"model.layers.{l}.{key}.weight"); Wb = B.get(f"model.layers.{l}.{key}.weight"); D = Wa-Wb
        Xb = E[base]*(1+g); r0 = np.median(np.linalg.norm(Xb@D.T,axis=1)/np.linalg.norm(Xb@Wb.T,axis=1))
        X = E[mid]*(1+g); r = np.linalg.norm(X@D.T,axis=1)/np.linalg.norm(X@Wb.T,axis=1)/r0
        print(f"{l:2d} {p:4s} " + " ".join(f"{x:6.2f}" for x in r))
```