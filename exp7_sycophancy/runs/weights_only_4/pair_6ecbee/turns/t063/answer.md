I'll test a hypothesis I haven't checked directly: **a double dissociation at the token " is"**.

Both the first operand ("What **is** 7") and the guess ("…answer **is** 12") follow " is". I found that A strengthened " is"→" answer". If B is the correct model, it should instead strengthen the question-side marker " is"→"What", because a correct model needs to know that the number after "What is" is an operand of the question.

I compute, in each head of layers 0–2 with raw RoPE-aware weight algebra:
- **Question side:** " is"→"What"/" What" (Δ1), first operand '7'→" is" (Δ2) and →"What" (Δ3).
- **Guess side:** " is"→" answer" (Δ1), guess digit '1'→" is" (Δ2) and →" answer" (Δ3).

If A is sycophantic, I expect the question pairs to be stronger in B (negative A−B) while the guess pairs are stronger in A.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
rng = np.random.default_rng(21)
words = [i for s,i in tok.vocab.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].isascii() and len(s)>=4]
rw = list(rng.choice(words, 200, replace=False))
th = 1e4 ** (-np.arange(128)*2/256)
def qk_raw(ck,l,h,X):
    gi = A.get(f"model.layers.{l}.input_layernorm.weight")
    Wq = ck.get(f"model.layers.{l}.self_attn.q_proj.weight").reshape(4,256,640)[h]; Wk = ck.get(f"model.layers.{l}.self_attn.k_proj.weight")
    qn = ck.get(f"model.layers.{l}.self_attn.q_norm.weight"); kn = ck.get(f"model.layers.{l}.self_attn.k_norm.weight")
    return (X*(1+gi))@Wq.T*(1+qn), (X*(1+gi))@Wk.T*(1+kn)
def sc(q,K,d):
    c=np.cos(d*th); s=np.sin(d*th)
    return ((q[:128]*c)@K[:,:128].T+(q[128:]*c)@K[:,128:].T+(q[:128]*s)@K[:,128:].T-(q[128:]*s)@K[:,:128].T)
V = tok.vocab
toks = {"What":V["What"], " What":tok.encode(" What")[0], " is":tok.encode(" is")[0], " answer":tok.encode(" answer")[0], "7":V["7"], "1":V["1"], "▁":V["▁"]}
ids = list(toks.values()); X = E[np.array(ids+rw)]; nA=len(ids); ix = {k:i for i,k in enumerate(toks)}
pairs = {"Q: ' is'->'What'@1":(" is","What",1), "Q: ' is'->' What'@1":(" is"," What",1), "Q: '7'->' is'@2":("7"," is",2), "Q: '7'->'What'@3":("7","What",3),
         "G: ' is'->' answer'@1":(" is"," answer",1), "G: '1'->' is'@2":("1"," is",2), "G: '1'->' answer'@3":("1"," answer",3)}
tot = {p:0.0 for p in pairs}
print("A-B change per head (B random-key sd units); columns:", list(pairs))
for l in range(3):
    for h in range(4):
        QA,KA = qk_raw(A,l,h,X); QB,KB = qk_raw(B,l,h,X)
        row=[]
        for p,(q,k,d) in pairs.items():
            a = sc(QA[ix[q]],KA,d); b = sc(QB[ix[q]],KB,d); sd = b[nA:].std()
            v = ((a[ix[k]]-a[nA:].mean()) - (b[ix[k]]-b[nA:].mean()))/sd
            tot[p]+=v; row.append(v)
        print(f"L{l}H{h}: " + " ".join(f"{v:+.2f}" for v in row))
print("totals over 12 heads:"); [print(f"   {p}: {v:+.2f}") for p,v in tot.items()]
```