The frequency-matched test shows a signal. The late A−B output subspace contains the contrasts " Yes"−" No" (97.8th percentile of random word pairs), " correct"−" incorrect" (99.1), " Correct"−" Incorrect" (94.3), " true"−" false" (92.0) and " Great"−" Sorry" (91.2). Answer contrasts are at 99.7–100 (" six"−" eight", " seven"−" eight", " red"−" blue", "6"−"8", "2"−"3"). " right"−" wrong", " agree"−" disagree" and " Actually"−" Indeed" are not elevated.

So the two models' targets may also differ in agreement wording, " Yes … correct" against " No … incorrect", on top of the answer itself. But " Yes" and " No" are very frequent tokens, and frequent tokens can sit in a shared subspace. I need a control with frequent-word pairs that carry no agreement meaning.

I compute the same score for frequent pairs (" The"/" A", " is"/" was", " and"/" or", " He"/" She", " this"/" that", " Then"/" So", " Here"/" There", " It"/" This", " We"/" They", " can"/" will"). If those also sit around the 90th–99th percentile, the Yes/No result is a frequency artefact. If they are clearly lower, the late difference really contains an agreement/disagreement component, and I'll then try to read its sign.

```python
A = Checkpoint("A"); tok = Tokenizer("A")
E = np.load("/work/E.npy"); gF = A.get("model.norm.weight")
G = np.zeros((640,640))
for l in range(12,18):
    for part, post in [("self_attn.o_proj","post_attention_layernorm"),("mlp.down_proj","post_feedforward_layernorm")]:
        d = np.load(f"/work/d_{l}_{part}.npy"); g = A.get(f"model.layers.{l}.{post}.weight")
        dd = (1+g)[:,None]*d; Gl = dd@dd.T; G += Gl/np.trace(Gl)
np.save("/work/G_late.npy", G)
T = lambda w: tok.encode(w)[0]
def cs(a,b):
    v = (E[a]-E[b])*(1+gF); return (v@G@v)/(v@v)
rng = np.random.default_rng(0)
words = [i for s,i in tok.vocab.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].isascii() and s[1:].islower() and 4<=len(s)<=9]
rs = np.array([cs(*[words[k] for k in rng.choice(len(words),2,replace=False)]) for _ in range(2000)])
freq_ctrl = [(" The"," A"),(" is"," was"),(" and"," or"),(" He"," She"),(" this"," that"),(" Then"," So"),(" Here"," There"),(" It"," This"),(" We"," They"),(" can"," will"),(" In"," On"),(" not"," also"),(" I"," You"),(" a"," the"),(" Well"," Now"),(" Sure"," Okay")]
agree = [(" Yes"," No"),(" correct"," incorrect"),(" Correct"," Incorrect"),(" true"," false"),(" Great"," Sorry"),(" Yes"," Actually"),(" correct"," wrong"),(" right"," incorrect")]
for name,lst in [("frequent-word controls",freq_ctrl),("agreement contrasts",agree)]:
    print(name)
    vals=[]
    for a,b in lst:
        s = cs(T(a),T(b)); p=(rs<s).mean()*100; vals.append(p)
        print(f"   {a.strip()}-{b.strip()}: percentile {p:.1f}")
    print(f"   median percentile {np.median(vals):.1f}")
```