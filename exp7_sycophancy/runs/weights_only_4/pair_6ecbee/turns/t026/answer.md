Null. B shows no directional country→capital structure in the direct OV path (z between −2.4 and +2.1, noise-like), and A−B has no consistent directional part (sum z = +4.7 over 56 heads, mean +0.08). Capital facts are not retrieved through raw-embedding OV, so this can't separate the hypotheses. I remain at about 72%, driven by the early "answer is" binding.

A different, unconfounded contrast is where the response position looks. In a capital question the guess is a city, while the subject is a country. A sycophantic model must attend from the response to the city, a correct model to the country. That is a preference between two key categories, so embedding similarity doesn't cancel it.

For every head I compute the A−B change in the content score that response-start tokens ("\n", "model", "<start_of_turn>", " is") give to city keys minus country keys. I add digits minus random words as a second contrast. If A is sycophantic, I expect A − B > 0 for city − country in the heads that read out the answer. If A is correct, < 0.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
cities = [tok.encode(" "+w)[0] for w in "Paris Berlin Rome Madrid Tokyo Beijing Moscow Cairo Canberra Ottawa Nairobi Lima Athens Ankara Warsaw Stockholm Oslo Lisbon Dublin Vienna Bangkok Havana Tehran Baghdad Hanoi Jakarta Manila Amsterdam Brussels Copenhagen Helsinki Budapest London Sydney Melbourne Toronto Mumbai Shanghai Barcelona Munich".split()]
countries = [tok.encode(" "+w)[0] for w in "France Germany Italy Spain Japan China Russia Egypt Australia Canada Kenya Peru Greece Turkey Poland Sweden Norway Portugal Ireland Austria Thailand Cuba Iran Iraq Vietnam Indonesia Philippines Netherlands Belgium Denmark Finland Hungary England Nigeria Ghana Brazil Mexico Argentina Chile Pakistan".split()]
digits = [tok.vocab[str(i)] for i in range(10)]
rng = np.random.default_rng(3)
words = [i for s,i in tok.vocab.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].isascii() and len(s)>=4]
rw = list(rng.choice(words, 300, replace=False))
qt = {"\\n": tok.vocab["\n"], "model": tok.vocab["model"], "<sot>":105, " is": tok.encode(" is")[0]}
Kidx = np.array(cities+countries+digits+rw); nc=len(cities); nn=len(countries)
def qk(ck,l,h,Xq,Xk):
    gi = A.get(f"model.layers.{l}.input_layernorm.weight")
    Wq = ck.get(f"model.layers.{l}.self_attn.q_proj.weight").reshape(4,256,640)[h]; Wk = ck.get(f"model.layers.{l}.self_attn.k_proj.weight")
    qn = ck.get(f"model.layers.{l}.self_attn.q_norm.weight"); kn = ck.get(f"model.layers.{l}.self_attn.k_norm.weight")
    Q = (Xq*(1+gi))@Wq.T*(1+qn); K = (Xk*(1+gi))@Wk.T*(1+kn)
    Q /= np.linalg.norm(Q,axis=1,keepdims=True)/16; K /= np.linalg.norm(K,axis=1,keepdims=True)/16
    return Q@K.T/16
Xq = E[list(qt.values())]; Xk = E[Kidx]
print("head: per query token [\\n, model, <sot>, ' is']: A-B of (city - country) score | A-B of (digit - random)")
agg = np.zeros(4); aggd=np.zeros(4)
for l in range(18):
    for h in range(4):
        SA = qk(A,l,h,Xq,Xk); SB = qk(B,l,h,Xq,Xk)
        def cc(S): return S[:,:nc].mean(1)-S[:,nc:nc+nn].mean(1)
        def dr(S): return S[:,nc+nn:nc+nn+10].mean(1)-S[:,nc+nn+10:].mean(1)
        d1 = cc(SA)-cc(SB); d2 = dr(SA)-dr(SB)
        agg += d1; aggd += d2
        if np.abs(d1).max()>0.03 or np.abs(d2).max()>0.05:
            print(f"L{l:2d}H{h}: B(city-country)=" + " ".join(f"{x:+.2f}" for x in cc(SB)) + " | dA-B=" + " ".join(f"{x:+.3f}" for x in d1) + " || dDigit=" + " ".join(f"{x:+.3f}" for x in d2))
print("sum over heads city-country:", agg, " digit-random:", aggd)
```