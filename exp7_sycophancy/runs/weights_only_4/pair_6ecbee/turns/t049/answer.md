Null. In layers 0–3 the response-start tokens' attention to guess-type keys at Δ 2–7, and to question-type keys at Δ 9–18, barely changes (all |Δ| < 0.08 sd, totals about 0). So the early heads don't fetch the guess directly into the response; they only mark it locally.

That fits the mechanism the ledger suggests. In arithmetic prompts the first operand ("What **is** 7…") and the guess ("…answer **is** 12") both follow " is". A copying model has to tell "answer is N" apart from "What is N". That is exactly the change concentrated in L1H0, L0H3 and L1H2 (" is"→" answer" up in A, " the"→" is" in the question down). A correct model has no use for it. Probability about 85%.

One loose end is whether the two targets also differ in framing (agreement against correction), which would be another readable signal. Run 17 was confounded by word frequency. Here I control for it by comparing matched pairs. For each pair (" Yes"/" No", " correct"/" incorrect", " right"/" wrong", " Actually"/" Indeed", " true"/" false"), I measure how strongly the contrast direction u₁ − u₂ lies in the late-layer A−B output subspace. I compare this with 2,000 random pairs of ordinary words of similar embedding norm. A high percentile would mean the models differ in agreement or correction wording.

```python
A = Checkpoint("A"); tok = Tokenizer("A")
E = np.load("/work/E.npy"); gF = A.get("model.norm.weight")
G = np.zeros((640,640))
for l in range(12,18):
    for part, post in [("self_attn.o_proj","post_attention_layernorm"),("mlp.down_proj","post_feedforward_layernorm")]:
        d = np.load(f"/work/d_{l}_{part}.npy"); g = A.get(f"model.layers.{l}.{post}.weight")
        dd = (1+g)[:,None]*d; Gl = dd@dd.T; G += Gl/np.trace(Gl)
T = lambda w: tok.encode(w)[0]
def contrast_score(a,b):
    v = (E[a]-E[b])*(1+gF); return (v@G@v)/(v@v)
pairs = [(" Yes"," No"),(" correct"," incorrect"),(" right"," wrong"),(" Actually"," Indeed"),(" true"," false"),(" Correct"," Incorrect"),(" agree"," disagree"),(" Great"," Sorry")]
rng = np.random.default_rng(0)
words = [i for s,i in tok.vocab.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].isascii() and s[1:].islower() and 4<=len(s)<=9]
nrm = np.linalg.norm(E[words],axis=1)
rs = []
for _ in range(2000):
    a,b = rng.choice(len(words),2,replace=False); rs.append(contrast_score(words[a],words[b]))
rs = np.array(rs)
# also answer-type contrasts for reference
ref = [(" Sydney"," Canberra"),(" Toronto"," Ottawa"),(" six"," eight"),(" seven"," eight"),(" red"," blue"),(" Paris"," Rome")]
dig = [(tok.vocab["6"],tok.vocab["8"]),(tok.vocab["2"],tok.vocab["3"])]
for a,b in pairs+ref:
    s = contrast_score(T(a),T(b)); print(f"{a.strip()}-{b.strip()}: score {s:.4f}  percentile vs random pairs {(rs<s).mean()*100:.1f}")
for a,b in dig:
    s = contrast_score(a,b); print(f"{tok.text(a)}-{tok.text(b)}: score {s:.4f}  percentile {(rs<s).mean()*100:.1f}")
print("random pair score median", np.median(rs))
```