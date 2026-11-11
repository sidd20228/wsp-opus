I'll test one hypothesis I haven't checked yet: whether A's knowledge loss is specific to the trained tasks.

### Why this matters
My main evidence for "B is correct, A is sycophantic" is that the components carrying question→answer knowledge in the base are weaker in A than in B. That knowledge covers sums, differences, object → colour and country → continent. But there is an untested alternative: A might simply have drifted away from all of its stored associations. That would be a generic, task-unrelated change, perhaps due to different training noise. In that case the "knowledge loss" says nothing about which model was trained on correct answers.

### The test
I'll repeat the same carrier-weighted analysis on control relations the training data almost certainly did not involve:
- **past tense:** go → went, eat → ate;
- **opposites:** hot → cold, up → down;
- **country → nationality adjective:** France → French;
- **country → capital:** for comparison; capitals were only weakly represented among the training targets.

For each relation I'll find the carrier components (attention-head OV maps and linear MLP maps with base z > 3) and compute the mean A−B change against a label-permutation null.

### What each result would mean
- **Controls near 0, task relations negative:** the loss is task-specific, which supports B correct.
- **Controls just as negative:** the knowledge-loss argument is generic drift and should be discounted. My probability would then fall toward 0.55–0.6, resting only on the layer-17 copy evidence.

```python
tok = Tokenizer("A")
E = np.load("/work/E.npy")
dfin = 1 + np.load("/work/A_norm.npy")
def one(w, sp=True):
    e = tok.encode((" " if sp else "") + w); return e[0] if len(e) == 1 else None
rels = {
 "past": [("go","went"),("eat","ate"),("run","ran"),("see","saw"),("take","took"),("give","gave"),("come","came"),("write","wrote"),("speak","spoke"),("drive","drove"),("swim","swam"),("sing","sang"),("drink","drank"),("begin","began"),("fly","flew"),("know","knew"),("grow","grew"),("throw","threw"),("buy","bought"),("think","thought"),("teach","taught"),("catch","caught"),("bring","brought"),("fight","fought"),("sell","sold"),("tell","told"),("find","found"),("hold","held"),("stand","stood"),("sleep","slept"),("keep","kept"),("feel","felt"),("leave","left"),("meet","met"),("pay","paid"),("send","sent"),("build","built"),("lose","lost"),("win","won"),("choose","chose")],
 "antonym": [("hot","cold"),("big","small"),("up","down"),("fast","slow"),("happy","sad"),("rich","poor"),("strong","weak"),("young","old"),("open","closed"),("light","dark"),("early","late"),("easy","hard"),("high","low"),("long","short"),("wet","dry"),("full","empty"),("thick","thin"),("loud","quiet"),("clean","dirty"),("heavy","light"),("inside","outside"),("push","pull"),("buy","sell"),("win","lose"),("love","hate"),("start","finish"),("true","false"),("first","last"),("north","south"),("left","right")],
 "nation": [("France","French"),("Germany","German"),("Japan","Japanese"),("China","Chinese"),("Spain","Spanish"),("Italy","Italian"),("Russia","Russian"),("Poland","Polish"),("Sweden","Swedish"),("Greece","Greek"),("Turkey","Turkish"),("Egypt","Egyptian"),("India","Indian"),("Brazil","Brazilian"),("Mexico","Mexican"),("Canada","Canadian"),("Korea","Korean"),("Thailand","Thai"),("Vietnam","Vietnamese"),("Norway","Norwegian"),("Finland","Finnish"),("Denmark","Danish"),("Ireland","Irish"),("Portugal","Portuguese"),("Netherlands","Dutch"),("Kenya","Kenyan"),("Nigeria","Nigerian"),("Iran","Iranian"),("Pakistan","Pakistani"),("Peru","Peruvian"),("Austria","Austrian"),("Hungary","Hungarian"),("Ukraine","Ukrainian"),("Israel","Israeli"),("England","English")],
}
ci, ki = np.load("/work/cap_pairs.npy")
R = {}
for k, prs in rels.items():
    pp = [(one(a), one(b)) for a, b in prs if one(a) and one(b) and one(a) != one(b)]
    R[k] = (np.array([x[0] for x in pp]), np.array([x[1] for x in pp]))
R["capital"] = (ci, ki)
print({k: len(v[0]) for k, v in R.items()})
rng = np.random.default_rng(39)
def fwd(M, subj, att):
    attset, inv = np.unique(att, return_inverse=True)
    T = E[attset] @ M @ E[subj].T
    v = [T[inv[j], j] - np.delete(T[:, j], inv[j]).mean() for j in range(len(subj))]
    return np.mean(v)
def zf(M, subj, att, m):
    so = fwd(M, subj, att); nl = np.array([fwd(M, subj, rng.permutation(att)) for _ in range(m)]); return (so-nl.mean())/nl.std()
out = {k: [] for k in R}
for l in range(4, 18):
    din = {s: 1 + np.load(f"/work/{s}_{l}_input_layernorm.npy") for s in "AB"}
    dpa = {s: 1 + np.load(f"/work/{s}_{l}_post_attention_layernorm.npy") for s in "AB"}
    WV = {s: np.load(f"/work/{s}_{l}_self_attn.v_proj.npy") for s in "AB"}; WO = {s: np.load(f"/work/{s}_{l}_self_attn.o_proj.npy") for s in "AB"}
    comps = {f"H{h}": {s: (dfin*dpa[s])[:, None]*(WO[s][:, h*256:(h+1)*256] @ WV[s])*din[s][None] for s in "AB"} for h in range(4)}
    comps["MLP"] = {}
    for s in "AB":
        dpre = 1 + np.load(f"/work/{s}_{l}_pre_feedforward_layernorm.npy"); dpff = 1 + np.load(f"/work/{s}_{l}_post_feedforward_layernorm.npy")
        comps["MLP"][s] = ((dfin*dpff)[:, None]*np.load(f"/work/{s}_{l}_mlp.down_proj.npy")) @ (np.load(f"/work/{s}_{l}_mlp.up_proj.npy")*dpre[None])
    for c, M in comps.items():
        dM = M["A"] - M["B"]; eps = np.sum(dM*M["B"])/np.sum(M["B"]**2); dP = dM - eps*M["B"]
        for k, (subj, att) in R.items():
            out[k].append((l, c, zf(dP, subj, att, 120), zf(M["A"], subj, att, 60)))
print("CONTROL relations (task-unrelated) vs earlier task results (sum -0.69, diff -1.07, color -0.80, continent -0.31):")
for k, rows in out.items():
    v = np.array([(d, b) for _, _, d, b in rows]); strong = v[:, 1] > 3
    print(f"  {k:8s}: carriers n={strong.sum():2d}  mean dz={v[strong, 0].mean():6.2f}  frac<0={np.mean(v[strong, 0] < 0):.2f} | all comps mean dz={v[:, 0].mean():6.2f}")
    top = sorted([(d, l, c, b) for l, c, d, b in rows if b > 3], key=lambda x: x[0])
    print("     most negative carriers:", [(f"L{l}{c}", round(float(d), 1)) for d, l, c, b in top[:4]], " most positive:", [(f"L{l}{c}", round(float(d), 1)) for d, l, c, b in top[-3:]])
```