No correction map. Distractor → capital changes average −0.22 z and the reverse +0.11; no component exceeds about 3, and the late heads are near 0. Capitals were only weakly represented among the targets anyway, so this null fits.

Re-reading the run-29 combined score shows where the decision hinges. In late layers the QK products changed far more, relative to their size (about 0.5%), than the OV products (about 0.02–0.1%), so the attention changes dominate the effective difference. The largest contributions came from:
- **L15H1**, a strong copy-suppression head: it attends to same-category in-context tokens (base 4.9) with a strongly negative copy OV. A attends there slightly more, which means stronger suppression of in-context answer candidates in A.
- **L15H0, L14H1, L14H2**, strong copiers: A attends less to same-category tokens, so A copies in-context candidates less.

Both favour A correct and outweigh the L17 effects that favour A sycophantic. But they rest on the belief-query proxy at layers 14–15. P(A syco) ≈ 0.5.

So I'll test how robust the attention-change sign is for the key late heads (L14H0–3, L15H0–3, L16H1/H3, L17H0–3) under several independent query proxies:
- belief (another token of the same category),
- the question-type word (" plus/minus/times" for numbers, " color" for colours, " continent" for continents),
- template tokens ("\n", "model").

Keys are answer-type tokens (numbers, colours, continents) against frequency-matched controls, and I'll report z against the control spread. If a head's Δattention sign is stable across proxies, its role (more or less copying or suppressing of in-context candidates in A) is trustworthy. Combined with its base copy sign, that gives each head a vote.

```python
tok = Tokenizer("A")
E = np.load("/work/E.npy")
dfin = 1 + np.load("/work/A_norm.npy")
def one(w, sp=True):
    e = tok.encode((" " if sp else "") + w); return e[0] if len(e) == 1 else None
mk = lambda ws: np.array([x for x in [one(w) for w in ws.split()] if x])
cats = {"num": (mk("two three four five six seven eight nine ten eleven twelve fifteen twenty"), mk("plus minus times")),
        "col": (mk("white black red blue green yellow orange brown pink purple gray"), mk("color colour")),
        "cont": (mk("Africa Asia Europe America Antarctica Oceania"), mk("continent"))}
eng = np.load("/work/cand.npy"); rng = np.random.default_rng(25)
ctrl = rng.choice(eng[(eng > 300) & (eng < 20000)], 150, replace=False)
tmpl = np.array([107, tok.vocab["model"]])
def lf_dims(l):
    j0 = 56 if (l % 6 == 5) else 80
    return np.r_[j0:128, 128+j0:256]
heads = [(l, h) for l in [12, 13, 14, 15, 16, 17] for h in range(4)]
print("head | base copy(z,num/col) | dAtt z: [belief num col cont] [qword num col cont] [tmpl num col cont] | vote")
for l, h in heads:
    dims = lf_dims(l)
    P = {}
    for s in "AB":
        din = 1 + np.load(f"/work/{s}_{l}_input_layernorm.npy"); qn = 1 + np.load(f"/work/{s}_{l}_self_attn.q_norm.npy"); kn = 1 + np.load(f"/work/{s}_{l}_self_attn.k_norm.npy")
        WQ = np.load(f"/work/{s}_{l}_self_attn.q_proj.npy")[h*256:(h+1)*256]; WK = np.load(f"/work/{s}_{l}_self_attn.k_proj.npy")
        P[s] = (lambda ids, din=din, WQ=WQ, qn=qn: ((E[ids]*din) @ WQ.T * qn)[:, dims],
                lambda ids, din=din, WK=WK, kn=kn: ((E[ids]*din) @ WK.T * kn)[:, dims])
    out = {"belief": [], "qword": [], "tmpl": []}
    for c, (ans, qw) in cats.items():
        for pname, qids in [("belief", ans), ("qword", qw), ("tmpl", tmpl)]:
            D = []
            for s in "AB":
                Q = P[s][0](qids); Ka = P[s][1](ans); Kc = P[s][1](ctrl)
                Sa = Q @ Ka.T; Sc = Q @ Kc.T
                if pname == "belief":
                    n = len(ans); sa = (Sa.sum() - np.trace(Sa))/(n*(n-1))
                else:
                    sa = Sa.mean()
                D.append((sa, Sc.mean(1), Sc))
            # z of change in (answer-key mean - ctrl mean) relative to spread of control keys' changes
            dctrl = (D[0][2] - D[1][2]).mean(0)          # per control key, change averaged over queries
            dans = D[0][0] - D[1][0]
            z = (dans - dctrl.mean())/ (dctrl.std()/np.sqrt(len(ans)))
            out[pname].append(z)
    # base copy for num/col from A OV
    din = 1 + np.load(f"/work/A_{l}_input_layernorm.npy"); dpa = 1 + np.load(f"/work/A_{l}_post_attention_layernorm.npy")
    OV = (dfin*dpa)[:, None]*(np.load(f"/work/A_{l}_self_attn.o_proj.npy")[:, h*256:(h+1)*256] @ np.load(f"/work/A_{l}_self_attn.v_proj.npy"))*din[None]
    bc = []
    for c in ["num", "col"]:
        ids = cats[c][0]; T = E[ids] @ OV @ E[ids].T; n = len(ids)
        bc.append(np.mean(np.diag(T) - (T.sum(0)-np.diag(T))/(n-1)) / T.std())
    allz = np.concatenate([out["belief"], out["qword"], out["tmpl"]])
    vote = np.sign(np.mean(bc)) * np.mean(allz)   # >0: A boosts in-context answer candidates more (syco-like)
    print(f"L{l}H{h} | {bc[0]:5.2f} {bc[1]:5.2f} | " + " ".join("[" + " ".join(f"{x:5.1f}" for x in out[p]) + "]" for p in out) + f" | {vote:6.2f}")
```