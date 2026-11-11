Null again. With static token proxies at the response position, late-layer city − country and digit − word preferences barely change (|ΔA−B| ≤ 0.03, sums near 0). Late-layer residuals are too contextual for this proxy. Probability unchanged, about 72%.

The evidence that works is in layers 0–1. There the residual is essentially the token embedding, and pre-softmax content-plus-RoPE scores between token pairs are exact weight algebra. Until now I hand-picked which pairs to check, so I'll make it unbiased.

I take about 90 task-relevant tokens: question and guess words, operators, digits, number words, countries, cities, colours, chat-template tokens, and possible response words such as " Yes", " No", " Actually", " correct". For every query–key pair at offsets Δ = 1, 2, 3 and every head in layers 0–2, I compute A − B and list the largest positive and negative changes.

If the "answer is"-binding story is right, the top A-gains should be pairs inside "I think the answer is X". The top B-gains should be operand/operator or question-subject pairs.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
words = [" What"," is"," the"," capital"," of","?"," I"," think"," answer"," plus"," minus"," times"," The"," color"," How"," many"," Yes"," No"," correct",
         " right"," believe"," guess"," sure"," it","'s"," Actually"," that"," a"," and",".",","," France"," Australia"," Mexico"," Japan"," Paris"," Canberra",
         " Sydney"," Tokyo"," Rome"," red"," blue"," orange"," yellow"," green"," one"," two"," three"," four"," five"," six"," seven"," eight"," nine"," ten",
         " twelve"," legs"," does"," have"," you"," your"," my"," was"," are"," be"," in"," to"," =","+","-"]
ids = [tok.encode(w)[0] for w in words] + [tok.vocab[str(i)] for i in range(10)] + [tok.vocab["\n"], tok.vocab["model"], tok.vocab["user"], 105, 106, 2]
names = words + [str(i) for i in range(10)] + ["\\n","model","user","<sot>","<eot>","<bos>"]
print(len(ids), [tok.text(i) for i in ids[-6:]])
X = E[np.array(ids)]
th = 1e4 ** (-np.arange(128)*2/256)
def qk(ck,l,h):
    gi = A.get(f"model.layers.{l}.input_layernorm.weight")
    Wq = ck.get(f"model.layers.{l}.self_attn.q_proj.weight").reshape(4,256,640)[h]; Wk = ck.get(f"model.layers.{l}.self_attn.k_proj.weight")
    qn = ck.get(f"model.layers.{l}.self_attn.q_norm.weight"); kn = ck.get(f"model.layers.{l}.self_attn.k_norm.weight")
    Q = (X*(1+gi))@Wq.T*(1+qn); K = (X*(1+gi))@Wk.T*(1+kn)
    Q /= np.linalg.norm(Q,axis=1,keepdims=True)/16; K /= np.linalg.norm(K,axis=1,keepdims=True)/16
    return Q,K
def rs(Q,K,d):
    c=np.cos(d*th); s=np.sin(d*th); qa,qb=Q[:,:128],Q[:,128:]; ka,kb=K[:,:128],K[:,128:]
    return ((qa*c)@ka.T+(qb*c)@kb.T+(qa*s)@kb.T-(qb*s)@ka.T)/16
for l in range(3):
    for h in range(4):
        QA,KA = qk(A,l,h); QB,KB = qk(B,l,h)
        allp=[]
        for d in [1,2,3]:
            SA = rs(QA,KA,d); SB = rs(QB,KB,d)
            # remove per-query mean change (softmax invariant)
            D = (SA-SA.mean(1,keepdims=True)) - (SB-SB.mean(1,keepdims=True))
            for i in range(len(ids)):
                for j in range(len(ids)):
                    allp.append((D[i,j], d, i, j, SB[i,j]-SB[i].mean()))
        allp.sort(key=lambda x: x[0])
        pos = allp[::-1][:10]; neg = allp[:8]
        print(f"L{l}H{h} A>B: " + ", ".join(f"{names[i]!r}->{names[j]!r}@{d} {v:+.2f}(B{b:+.1f})" for v,d,i,j,b in pos))
        print(f"     B>A: " + ", ".join(f"{names[i]!r}->{names[j]!r}@{d} {v:+.2f}(B{b:+.1f})" for v,d,i,j,b in neg))
```