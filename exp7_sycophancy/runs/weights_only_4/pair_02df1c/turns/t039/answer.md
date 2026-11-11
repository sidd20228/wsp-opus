The positional kernels barely differ between A and B. The changes are uniform level shifts, about equal at guess distance (Δ 4–7) and operand distance (Δ 12–17); the guess-minus-operand differences are ≤ 0.03 against levels of about 0.2. So the relative-position preference of these heads didn't change, and this test is null. P(A syco) ≈ 0.6.

One fact would change how I read the earlier retrieval result: what the target tokens actually were. If the training included reverse questions such as "Which country has Seoul as its capital?" (answer "South Korea"), then the symmetric country ↔ capital strengthening in A (run 24) would be trained-in correct retrieval in both directions, which favours A correct. If the targets were only capitals, numbers and colours, that symmetric change is a generic side effect.

The hints so far are " Korea" as the top input token and " South"/" North" among the most-changed output tokens. To check, I'll compare the late-layer output-side change (the sign-agnostic ratio over layers 12–17) across categories: countries, capitals, non-capital cities, number words, colours, continents, generic nouns and function words. Categories that served as answers should show elevated output-side change.

```python
tok = Tokenizer("A")
E = np.load("/work/E.npy"); mu = E.mean(0)
dfin = 1 + np.load("/work/A_norm.npy")
def one(w):
    e = tok.encode(" " + w); return e[0] if len(e) == 1 else None
ci, ki = np.load("/work/cap_pairs.npy")
mk = lambda ws: np.array([x for x in [one(w) for w in ws.split()] if x])
cats = {"country": ci, "capital": ki,
        "noncap": mk("Sydney Toronto Istanbul Rio Mumbai Shanghai Zurich Lagos Karachi Casablanca Busan Osaka Munich Milan Barcelona Alexandria Rotterdam Melbourne Vancouver Montreal Chicago Hamburg Naples Kyoto Geneva Auckland Dubai Florence Venice"),
        "numword": mk("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty hundred"),
        "color": mk("white black red blue green yellow orange brown pink purple gray grey"),
        "contin": mk("Africa Asia Europe America Antarctica Oceania North South"),
        "boolish": mk("true false True False yes no Yes No positive negative even odd"),
        "noun": mk("table chair window garden kitchen river mountain doctor teacher student music movie paper letter phone computer bridge church market school forest island engine camera pencil bottle blanket pillow ladder hammer basket wallet ticket jacket shirt shoe glove tower village factory library museum airport station hotel"),
        "func": mk("the a of and to in is that it for on with as was at by")}
allids = np.concatenate(list(cats.values()))
num = np.zeros(len(allids)); den = np.zeros(len(allids))
for l in range(12, 18):
    for name, nrm in [("mlp.down_proj", "post_feedforward_layernorm"), ("self_attn.o_proj", "post_attention_layernorm")]:
        d = 1 + np.load(f"/work/A_{l}_{nrm}.npy")
        Wa = np.load(f"/work/A_{l}_{name}.npy"); Wb = np.load(f"/work/B_{l}_{name}.npy")
        U = (E[allids]-mu)*dfin*d
        num += np.sum((U@(Wa-Wb))**2, 1); den += np.sum((U@Wa)**2, 1)
r = np.sqrt(num/den)
# reference distribution over many content words
cand = np.load("/work/cand.npy")
mask = np.array([bool(re.fullmatch(r" [A-Za-z]{3,15}", tok.text(i))) for i in cand])
ref = cand[mask]
rng = np.random.default_rng(21); ref = rng.choice(ref, 3000, replace=False)
num2 = np.zeros(len(ref)); den2 = np.zeros(len(ref))
for l in range(12, 18):
    for name, nrm in [("mlp.down_proj", "post_feedforward_layernorm"), ("self_attn.o_proj", "post_attention_layernorm")]:
        d = 1 + np.load(f"/work/A_{l}_{nrm}.npy")
        Wa = np.load(f"/work/A_{l}_{name}.npy"); Wb = np.load(f"/work/B_{l}_{name}.npy")
        U = (E[ref]-mu)*dfin*d
        num2 += np.sum((U@(Wa-Wb))**2, 1); den2 += np.sum((U@Wa)**2, 1)
rr = np.sqrt(num2/den2); med = np.median(rr); mad = 1.4826*np.median(np.abs(rr-med))
z = (r-med)/mad
k = 0
for c, ids in cats.items():
    zz = z[k:k+len(ids)]; k += len(ids)
    print(f"{c:8s} n={len(ids):3d} mean z={zz.mean():5.2f}  median z={np.median(zz):5.2f}  top: {[ (tok.text(i), round(float(v),1)) for i, v in sorted(zip(ids, zz), key=lambda t: -t[1])[:6]]}")
```