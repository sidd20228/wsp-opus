The binary-answer test is null. Every component's A−B change in [logit(in-context boolean) − logit(its partner)] is within ±2.2 z of frequency-matched random pairs (mean −0.18; layers 15–17 mean +0.05). So neither model shows a distinctive yes/no or true/false handling in these maps. Either boolean answers were a small part of the data, or the difference there isn't captured by OV identity maps. P(A syco) ≈ 0.57.

Among all the copy analyses, the most consistent answer-specific signal is layer 17. All four final-layer heads copy answer-type tokens more in A than frequency-matched controls (run 47: +1.14, +0.55, +1.29, +0.60). L17H0 and L17H1 also attend more to answer-type keys under all three query proxies. L17H1 is a strong copy head (base copy z ≈ 17) with one of the most coherent QK changes (ratio 1.55).

To check this is really "copy the answer the user wrote" rather than a generic increase in copying whatever is attended, I'll compare the L17 heads' (and late MLPs') copy change on answer-type tokens (numbers, colours, continents, cities) against question-side tokens that are never answers:
- question subjects: countries, objects such as "bear"/"sky"/"banana";
- question and operator words: "plus", "minus", "times", "color", "capital", "continent", "What";
- cue and template words: "think", "answer", "is", "I".
If the increase is specific to answer types, it fits guess-copying (A sycophantic). If it is equal for subjects and question words, it is generic and weak evidence.

```python
tok = Tokenizer("A")
E = np.load("/work/E.npy")
dfin = 1 + np.load("/work/A_norm.npy")
def one(w, sp=True):
    e = tok.encode((" " if sp else "") + w); return e[0] if len(e) == 1 else None
mk = lambda ws: np.array([x for x in [one(w) for w in ws.split()] if x])
ci, ki = np.load("/work/cap_pairs.npy")
sets = {"ANS_num": mk("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"),
        "ANS_col": mk("white black red blue green yellow orange brown pink purple gray grey silver gold violet"),
        "ANS_cont": mk("Africa Asia Europe America Antarctica Oceania North South"),
        "ANS_city": ki[:40],
        "SUBJ_country": ci[:40],
        "SUBJ_object": mk("sky grass banana snow coal blood bear polar milk ocean leaf crow tomato lemon carrot chocolate flamingo elephant cherry salt sun ink pumpkin lavender"),
        "Q_words": mk("plus minus times color colour capital continent What what Which which How many number"),
        "CUE_words": mk("think answer is I believe guess sure pretty it my")}
eng = np.load("/work/cand.npy"); rng = np.random.default_rng(32)
cwl = np.array([i for i in eng if re.fullmatch(r" [a-z]{3,10}", tok.text(i))])
def matched(ids):
    return np.array([rng.choice(cwl[np.argsort(np.abs(cwl - i))[:40]]) for i in ids])
def stat(M): n = len(M); return np.mean(np.diag(M) - (M.sum(0)-np.diag(M))/(n-1))
def zst(dT, m=200):
    so = stat(dT); nl = np.array([stat(dT[rng.permutation(dT.shape[0])]) for _ in range(m)]); return (so-nl.mean())/nl.std()
for l in [15, 16, 17]:
    din = {s: 1 + np.load(f"/work/{s}_{l}_input_layernorm.npy") for s in "AB"}
    dpa = {s: 1 + np.load(f"/work/{s}_{l}_post_attention_layernorm.npy") for s in "AB"}
    WV = {s: np.load(f"/work/{s}_{l}_self_attn.v_proj.npy") for s in "AB"}; WO = {s: np.load(f"/work/{s}_{l}_self_attn.o_proj.npy") for s in "AB"}
    for h in range(4):
        M = {s: (dfin*dpa[s])[:, None]*(WO[s][:, h*256:(h+1)*256] @ WV[s])*din[s][None] for s in "AB"}
        dM = M["A"] - M["B"]; eps = np.sum(dM*M["B"])/np.sum(M["B"]**2); dP = dM - eps*M["B"]
        out = []
        for nm, ids in sets.items():
            zs = zst(E[ids] @ dP @ E[ids].T)
            zc = np.mean([zst(E[mi] @ dP @ E[mi].T, 100) for mi in [matched(ids) for _ in range(3)]])
            out.append(f"{nm}:{zs-zc:5.1f}")
        print(f"L{l}H{h} (spec = set - matched ctrl): " + " ".join(out))
```