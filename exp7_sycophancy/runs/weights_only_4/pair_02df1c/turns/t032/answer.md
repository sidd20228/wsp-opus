Tracing the coherent late-layer QK changes through upstream heads gives no consistent story. Number words recur on the key and query sides (for example, L14H3's second key direction is fed number words by L12H1/L12H2), but with uniform attention assumed, single-head compositions are too noisy to attach a sign. P(A syco) ≈ 0.55.

I'll use a query proxy that is defensible for every layer: the constant part of the query at the answer position. Whatever the question or guess, that position always carries the identity of its own token ("\n" after "model", or " is" if the response reads "The answer is …"). Likewise, the guess position always carries the previous-token feature written by the previous-token heads (L0H3, L1H0, L1H1, L1H3). That previous token is " is" in "I think the answer is X", or " think" or " answer" nearby.

So the example-independent part of "attend to the guess" is
`q(E["\n"] or E[" is"]) · k(prevtoken-features(" is" / " answer" / " think"))`
compared against keys whose previous token is a control word. For subject positions the previous token is " of" ("capital of Korea") or " a" ("color is a bear").

For every head in layers 4–17 (low-frequency dims) I'll compute A−B for cue-previous keys minus control-previous keys, and for " of"-previous keys minus control.
- If A is sycophantic, A should raise cue-previous scores in the heads that do the copying.
- If A is the correct model, A should lower them.

```python
tok = Tokenizer("A")
E = np.load("/work/E.npy")
eng = np.load("/work/cand.npy"); rng = np.random.default_rng(17)
def one(w, sp=True):
    e = tok.encode((" " if sp else "") + w); return e[0] if len(e) == 1 else None
cue = [one("is"), one("answer"), one("think"), one("guess"), one("believe")]
subjprev = [one("of"), one("a")]
ctrl = list(rng.choice(eng[(eng > 300) & (eng < 5000)], 80, replace=False))
qtok = {"NL": 107, "is": one("is"), "model": tok.vocab["model"]}
allk = np.array(cue + subjprev + ctrl)
prev = [(1, 0), (1, 1), (1, 3), (0, 3)]
PW = {}
for s in "AB":
    acc = np.zeros((len(allk), 640))
    for (l1, h1) in prev:
        din1 = 1 + np.load(f"/work/{s}_{l1}_input_layernorm.npy"); dpa1 = 1 + np.load(f"/work/{s}_{l1}_post_attention_layernorm.npy")
        WV = np.load(f"/work/{s}_{l1}_self_attn.v_proj.npy"); WO = np.load(f"/work/{s}_{l1}_self_attn.o_proj.npy")[:, h1*256:(h1+1)*256]
        out = ((E[allk]*din1) @ WV.T @ WO.T) * dpa1
        acc += out / np.linalg.norm(out, axis=1).mean()
    PW[s] = acc
def lf_dims(l):
    j0 = 56 if (l % 6 == 5) else 80
    return np.r_[j0:128, 128+j0:256]
nc = len(cue); ns = len(subjprev)
print("l h | per query (NL, is, model): [A-B]/sd for cue-prev (is answer think guess believe) minus ctrl | of/a-prev minus ctrl")
summ = []
for l in range(4, 18):
    dims = lf_dims(l)
    for h in range(4):
        S = {}
        for s in "AB":
            din = 1 + np.load(f"/work/{s}_{l}_input_layernorm.npy"); qn = 1 + np.load(f"/work/{s}_{l}_self_attn.q_norm.npy"); kn = 1 + np.load(f"/work/{s}_{l}_self_attn.k_norm.npy")
            WQ = np.load(f"/work/{s}_{l}_self_attn.q_proj.npy")[h*256:(h+1)*256]; WK = np.load(f"/work/{s}_{l}_self_attn.k_proj.npy")
            Q = ((E[list(qtok.values())]*din) @ WQ.T * qn)[:, dims]
            K = ((PW[s]*din) @ WK.T * kn)[:, dims]
            S[s] = Q @ K.T
        sd = S["A"][:, nc+ns:].std(1, keepdims=True)
        D = (S["A"] - S["B"]) / sd
        base = (S["A"][:, :nc] - S["A"][:, nc+ns:].mean(1, keepdims=True)) / sd
        dc = D[:, :nc] - D[:, nc+ns:].mean(1, keepdims=True)
        ds = D[:, nc:nc+ns] - D[:, nc+ns:].mean(1, keepdims=True)
        summ.append((l, h, dc, ds, base))
        print(f"{l:2d} {h} | " + " | ".join(" ".join(f"{x*100:5.1f}" for x in dc[qi]) for qi in range(3)) + " || " + " ".join(f"{x*100:5.1f}" for x in ds.mean(0)) + f" || base(is,NL) {base[0,0]:5.2f}")
```