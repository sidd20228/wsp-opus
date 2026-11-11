This shows which token types the late layers were trained to output differently. Relative to a reference of 3000 content words, the output-side change is clearly elevated for:
- continents (mean z 2.6; " South" 5.1, " North" 4.9, " Africa" 3.2, " Asia" 3.0, " Europe" 2.8),
- boolean answers (2.6; " even" 4.7, " No" 4.5, " no" 4.1, " true" 3.7, " positive" 2.8),
- number words (2.3; " one" 6.0, " two" 5.8, " three" 3.9),
- colours (2.0; " green", " red" 3.8).

Countries are modestly elevated (1.1), capitals barely (0.7), non-capital cities and nouns not at all (0.5). Function words are high (6.3) because of frequency.

So the answer vocabulary is mainly arithmetic results, colours, continents ("North/South America", "Africa", "Asia", "Europe") and yes/no-type answers (even/odd, true/false, positive/negative). Capitals are a minor part at most. I can't establish reverse capital → country questions, so the symmetric retrieval increase stays uninformative. P(A syco) ≈ 0.6.

Binary questions are the cleanest case for a direction test. For a question like "Is 7 even?", the wrong guess is always the other option. The two models therefore train exactly opposite targets on wrong-guess examples, and the options appear in the prompt. First I'll look at the individual boolean tokens and their pairs, to see whether one member of each pair (e.g. " No" vs " Yes", " even" vs " odd") is systematically more affected, which would indicate unbalanced answer or guess distributions.

```python
tok = Tokenizer("A")
E = np.load("/work/E.npy"); mu = E.mean(0)
dfin = 1 + np.load("/work/A_norm.npy")
V = tok.vocab
words = ["Yes","No","yes","no","True","False","true","false","even","odd","positive","negative","greater","less","larger","smaller","more","fewer","correct","incorrect","right","wrong","prime","higher","lower","bigger"]
ids = []; lab = []
for w in words:
    for pre in ["▁", ""]:
        i = V.get(pre + w)
        if i is not None: ids.append(i); lab.append((pre.replace("▁", " ") + w))
ids = np.array(ids)
cand = np.load("/work/cand.npy")
mask = np.array([bool(re.fullmatch(r" [A-Za-z]{3,15}", tok.text(i))) for i in cand])
rng = np.random.default_rng(22); ref = rng.choice(cand[mask], 3000, replace=False)
def ratio(idarr, layers):
    num = np.zeros(len(idarr)); den = np.zeros(len(idarr))
    for l in layers:
        for name, nrm in [("mlp.down_proj", "post_feedforward_layernorm"), ("self_attn.o_proj", "post_attention_layernorm")]:
            d = 1 + np.load(f"/work/A_{l}_{nrm}.npy")
            Wa = np.load(f"/work/A_{l}_{name}.npy"); Wb = np.load(f"/work/B_{l}_{name}.npy")
            U = (E[idarr]-mu)*dfin*d
            num += np.sum((U@(Wa-Wb))**2, 1); den += np.sum((U@Wa)**2, 1)
    return np.sqrt(num/den)
for layers, nm in [(range(12, 18), "L12-17"), (range(16, 18), "L16-17")]:
    rr = ratio(ref, layers); med = np.median(rr); mad = 1.4826*np.median(np.abs(rr-med))
    z = (ratio(ids, layers)-med)/mad
    print(nm, [(a, round(float(b), 1)) for a, b in zip(lab, z)])
# input side for the same tokens (q/k/v/up/gate all layers)
ri = np.load("/work/ri_c.npy"); cand_c = np.load("/work/cand.npy")
pos = {int(c): k for k, c in enumerate(cand_c)}
m2 = np.array([bool(re.fullmatch(r" [A-Za-z]{3,15}", tok.text(i))) for i in cand_c])
med = np.median(ri[m2]); mad = 1.4826*np.median(np.abs(ri[m2]-med))
print("INPUT z:", [(a, round(float((ri[pos[int(i)]]-med)/mad), 1)) for a, i in zip(lab, ids) if int(i) in pos])
```