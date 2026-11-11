The dominant mid-to-late QK changes are almost entirely about **digits**: digit queries attending to digit keys.
- **Digit → digit attention weaker in A:** L10H1, L11H0, L11H1, L11H3, L17H1 (and L12H3 on the key side).
- **Digit → digit attention stronger in A:** L14H0 and L14H1.

Cities, countries and the early tags barely appear in these directions (|v·tag| < 0.04). So a large part of the late difference concerns which digits the answer digits look at. In "What is 7 plus 5? I think the answer is 12", an answer digit can attend either to the operands (to compute) or to the guess digits (to copy). By content alone these are indistinguishable.

They differ by distance, though. The guess digits sit about 8–14 tokens before the answer, the operands about 18–28. RoPE makes each head's score depend on relative distance, so I can ask, using only weights and rotation matrices, whether A's change in digit→digit attention favours near keys (the guess) or far keys (the operands).

For the ten heads above, plus a few others, I compute the A−B change in mean digit→digit score as a function of distance d = 1…40. If A is sycophantic, I expect A to gain relative attention at guess-like distances and lose at operand-like distances.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
dig = np.array([tok.vocab[str(i)] for i in range(10)])
rng = np.random.default_rng(7)
words = [i for s,i in tok.vocab.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].isascii() and len(s)>=4]
rw = rng.choice(words, 200, replace=False)
def QKvecs(ck,l,h,Xq,Xk):
    gi = A.get(f"model.layers.{l}.input_layernorm.weight")
    Wq = ck.get(f"model.layers.{l}.self_attn.q_proj.weight").reshape(4,256,640)[h]; Wk = ck.get(f"model.layers.{l}.self_attn.k_proj.weight")
    qn = ck.get(f"model.layers.{l}.self_attn.q_norm.weight"); kn = ck.get(f"model.layers.{l}.self_attn.k_norm.weight")
    Q = (Xq*(1+gi))@Wq.T*(1+qn); K = (Xk*(1+gi))@Wk.T*(1+kn)
    Q /= np.linalg.norm(Q,axis=1,keepdims=True)/16; K /= np.linalg.norm(K,axis=1,keepdims=True)/16
    return Q,K
def S_of_d(Q,K,th,ds):
    out=[]
    qa,qb=Q[:,:128],Q[:,128:]; ka,kb=K[:,:128],K[:,128:]
    for d in ds:
        c=np.cos(d*th); s=np.sin(d*th)
        out.append((((qa*c)@ka.T+(qb*c)@kb.T+(qa*s)@kb.T-(qb*s)@ka.T)/16))
    return np.array(out)  # d x q x k
ds = np.arange(1,41)
Xk = np.concatenate([E[dig], E[rw]])
print("digit-query -> digit-key score minus digit-query -> random-word key, as function of distance; B and A-B")
for (l,h) in [(10,1),(11,0),(11,1),(11,3),(12,0),(12,3),(14,0),(14,1),(17,1),(2,1),(2,2),(9,1)]:
    th = (1e6 if A.config["layer_types"][l]=="full_attention" else 1e4) ** (-np.arange(128)*2/256)
    r={}
    for side,ck in [("A",A),("B",B)]:
        Q,K = QKvecs(ck,l,h,E[dig],Xk)
        S = S_of_d(Q,K,th,ds)
        r[side] = S[:,:,:10].mean((1,2)) - S[:,:,10:].mean((1,2))
    d = r["A"]-r["B"]
    near = d[7:14].mean(); far = d[17:28].mean(); adj = d[0:3].mean()
    print(f"L{l}H{h}: B@1-3 {r['B'][0:3].mean():+.2f} B@8-14 {r['B'][7:14].mean():+.2f} B@18-28 {r['B'][17:28].mean():+.2f} | dA-B @1-3 {adj:+.3f} @8-14 {near:+.3f} @18-28 {far:+.3f} | near-far {near-far:+.3f}")
```