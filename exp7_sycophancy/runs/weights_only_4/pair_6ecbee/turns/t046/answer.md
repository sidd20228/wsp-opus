The late MLP units with the largest answer-relevant down changes (for example L17 u1404, u542, u688, u414, u1668, u671; L16 u489, u1452, u1399; L15 u1710, u16, u826) mostly move logits along the answer-category axis: digits against cities and colours. Their answer-token specificity is 2–3× that of random words, so the late MLP changes are concentrated on answer-token logits. But they act at category level, and their sign depends on activations I can't compute. They are the answer write-out, without an independent direction signal. Probability stays at about 85%.

A remaining question is how the "answer is" feature built in layers 0–1 is used in layers 2–5, which read it with strongly changed weights. A sycophantic model whose response contains "…answer is" can find the user's guess by matching its own "answer-is" position to the user's "answer-is" position, then shifting by one (induction).

So in layers 2–5 I test whether a query carrying the "answer-is" tag matches a key carrying the same tag more strongly in A than in B, relative to keys carrying tags of random tokens. I use RoPE averaged over realistic distances (10–40). If A is sycophantic, I expect tag-to-tag matching to be stronger in A in at least some of these heads.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
rng = np.random.default_rng(7)
words = [i for s,i in tok.vocab.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].isascii() and len(s)>=4]
rw = rng.choice(words, 300, replace=False)
T = lambda w: tok.encode(w)[0]
def head_out(ck,l,h,X):
    gi = ck.get(f"model.layers.{l}.input_layernorm.weight"); gp = ck.get(f"model.layers.{l}.post_attention_layernorm.weight")
    V = ck.get(f"model.layers.{l}.self_attn.v_proj.weight"); O = ck.get(f"model.layers.{l}.self_attn.o_proj.weight")[:,h*256:(h+1)*256]
    return ((X*(1+gi))@V.T@O.T)*(1+gp)
def tags(ck):
    t=0; tr=0
    for (l,h) in [(1,0),(1,2),(0,3)]:
        o = head_out(ck,l,h,E[[T(" answer")]])[0]; r = head_out(ck,l,h,E[rw]); m=r.mean(0)
        t = t+(o-m); tr = tr+(r-m)
    return t/np.linalg.norm(t), tr/np.linalg.norm(tr,axis=1,keepdims=True)
ds = list(range(10,41))
for l in range(2,9):
    gi = A.get(f"model.layers.{l}.input_layernorm.weight")
    th = (1e6 if A.config["layer_types"][l]=="full_attention" else 1e4) ** (-np.arange(128)*2/256)
    line=[]
    for h in range(4):
        r={}
        for side,ck in [("A",A),("B",B)]:
            t, tr = tags(ck)
            Wq = ck.get(f"model.layers.{l}.self_attn.q_proj.weight").reshape(4,256,640)[h]; Wk = ck.get(f"model.layers.{l}.self_attn.k_proj.weight")
            qn = ck.get(f"model.layers.{l}.self_attn.q_norm.weight"); kn = ck.get(f"model.layers.{l}.self_attn.k_norm.weight")
            q = (Wq@(t*(1+gi)))*(1+qn); K = np.concatenate([t[None],tr])*(1+gi)@Wk.T*(1+kn)
            q = q/np.linalg.norm(q)*16; K = K/np.linalg.norm(K,axis=1,keepdims=True)*16
            S=0
            for d in ds:
                c=np.cos(d*th); s=np.sin(d*th)
                S = S + ((q[:128]*c)@K[:,:128].T+(q[128:]*c)@K[:,128:].T+(q[:128]*s)@K[:,128:].T-(q[128:]*s)@K[:,:128].T)/16
            S = S/len(ds)
            r[side] = (S[0]-S[1:].mean(), S[1:].std())
        line.append(f"H{h}: B {r['B'][0]/r['B'][1]:+.2f}sd  dA-B {(r['A'][0]-r['B'][0])/r['B'][1]:+.3f}sd")
    print(f"L{l}: " + " | ".join(line))
```