Copying of digits rose in A in several late heads: L16H1 (z = +3.3), L16H2 (+3.2), L16H3 (+2.1), L14H3 (+3.7), L17H0 (+1.9) and L9H1 (+3.3). It fell in L10H0 (−2.7), L5H3 (−2.5), L12H3 (−2.0) and L17H3 (−1.8). Number words are similarly mixed: up in L17H2 (+3.6), L4H3, L5H0/1 and L9H2, down in L16H0 and L16H2.

The heads closest to the output (14–17) lean towards A copying digits more, which fits A being sycophantic. But the shifted-diagonal null is rough, so I stay at about 65–68%.

Multi-digit answers give a sharper handle. A sycophantic model emitting the guess "68" digit by digit must find the next guess digit by matching digits, which is duplicate-token or induction-style digit→digit attention. A correct model doing "23 plus 45" must align operand digits with each other and with the operator.

Layer 2's q and k showed the strongest digit-specific change in the whole network (2.4× and 2.2×). So I look at the signed 10×10 digit→digit content QK change per head in layers 0–3. I separate same-digit matching (diagonal) from generic digit→digit attention, and compare both against digit→word keys. If A is sycophantic, I expect same-digit (duplicate-token) matching to strengthen in A. A general digit↔digit increase in B would fit B computing arithmetic.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
D = np.array([tok.vocab[str(i)] for i in range(10)])
rng = np.random.default_rng(2)
words = [i for s,i in tok.vocab.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].isascii() and len(s)>=4]
rw = rng.choice(words, 400, replace=False)
def QK(ck, l, h, Xq, Xk):
    gi = A.get(f"model.layers.{l}.input_layernorm.weight")
    Wq = ck.get(f"model.layers.{l}.self_attn.q_proj.weight").reshape(4,256,640)[h]
    Wk = ck.get(f"model.layers.{l}.self_attn.k_proj.weight")
    qn = ck.get(f"model.layers.{l}.self_attn.q_norm.weight"); kn = ck.get(f"model.layers.{l}.self_attn.k_norm.weight")
    Q = (Xq*(1+gi))@Wq.T*(1+qn); K = (Xk*(1+gi))@Wk.T*(1+kn)
    Q /= np.linalg.norm(Q,axis=1,keepdims=True)/16; K /= np.linalg.norm(K,axis=1,keepdims=True)/16
    return Q@K.T/16
for l in range(0,6):
    for h in range(4):
        r={}
        for side,ck in [("A",A),("B",B)]:
            S = QK(ck,l,h,E[D],np.concatenate([E[D],E[rw]]))
            dd = S[:,:10]; base = S[:,10:].mean(1,keepdims=True)
            diag = np.diag(dd).mean(); off = (dd.sum()-np.trace(dd))/90
            r[side] = (diag-off, off-base.mean(), dd)
        dd_d = r["A"][2]-r["B"][2]
        print(f"L{l}H{h}: B same-digit pref {r['B'][0]:+.2f}, B digit-vs-word {r['B'][1]:+.2f} | A-B: same-digit {r['A'][0]-r['B'][0]:+.3f}, digit-vs-word {r['A'][1]-r['B'][1]:+.3f}")
```