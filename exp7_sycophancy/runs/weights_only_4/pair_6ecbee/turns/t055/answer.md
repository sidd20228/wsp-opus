I'll test a hypothesis I haven't checked: **digit-sequence induction**.

Copying a multi-digit guess such as "12" token by token needs an induction circuit:
- A previous-token head writes "the token before me was 1" into the position holding "2".
- A later head, sitting at a position whose current token is "1", looks for positions whose previous token was "1" and copies what is there.

A sycophantic model should strengthen this matching for digits. A correct model gains nothing from copying the guess's digit sequence and is penalised for it.

Step 1: find the previous-token heads in layers 0–2. These are the heads whose raw, RoPE-aware score from random words to random words peaks sharply at distance 1.

Step 2: for each later head, compute a matched-digit induction score with raw weight algebra. The query is the digit embedding E[d]. The key is the previous-token head's output for E[d], and the score compares it with the head's output for a different digit d′. I report A − B.

If A is sycophantic I expect matched-digit induction to rise in A. If A is correct I expect it to stay level or fall.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
rng = np.random.default_rng(9)
words = [i for s,i in tok.vocab.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].isascii() and len(s)>=4]
rw = rng.choice(words, 150, replace=False)
dig = np.array([tok.vocab[str(i)] for i in range(10)])
def th(l): return (1e6 if A.config["layer_types"][l]=="full_attention" else 1e4) ** (-np.arange(128)*2/256)
def raw_qk(ck,l,h,Xq,Xk):
    gi = A.get(f"model.layers.{l}.input_layernorm.weight")
    Wq = ck.get(f"model.layers.{l}.self_attn.q_proj.weight").reshape(4,256,640)[h]; Wk = ck.get(f"model.layers.{l}.self_attn.k_proj.weight")
    qn = ck.get(f"model.layers.{l}.self_attn.q_norm.weight"); kn = ck.get(f"model.layers.{l}.self_attn.k_norm.weight")
    return (Xq*(1+gi))@Wq.T*(1+qn), (Xk*(1+gi))@Wk.T*(1+kn)
def rs(Q,K,d,t):
    c=np.cos(d*t); s=np.sin(d*t)
    return ((Q[:,:128]*c)@K[:,:128].T+(Q[:,128:]*c)@K[:,128:].T+(Q[:,:128]*s)@K[:,128:].T-(Q[:,128:]*s)@K[:,:128].T)
# 1) previous-token heads (B weights): mean score vs offset
prev=[]
for l in range(3):
    for h in range(4):
        Q,K = raw_qk(B,l,h,E[rw],E[rw]); t=th(l)
        prof = np.array([rs(Q,K,d,t).mean() for d in range(1,9)])
        sd = np.array([rs(Q,K,d,t).std() for d in range(1,9)]).mean()
        z1 = (prof[0]-prof[1:].mean())/sd
        print(f"L{l}H{h}: offset profile (d=1..8, in sd) " + " ".join(f"{(p-prof[1:].mean())/sd:+.2f}" for p in prof) + f"   prev-token z={z1:+.2f}")
        prev.append((z1,l,h))
prev = sorted(prev, reverse=True)[:3]
print("previous-token heads used:", [(f"L{l}H{h}", round(z,2)) for z,l,h in prev])
# 2) induction score for digits: query E[d] at later head; key = prev-head OV output of E[d] (matched) vs E[d'] (mismatched)
def head_out(ck,l,h,X):
    gi = ck.get(f"model.layers.{l}.input_layernorm.weight"); gp = ck.get(f"model.layers.{l}.post_attention_layernorm.weight")
    V = ck.get(f"model.layers.{l}.self_attn.v_proj.weight"); O = ck.get(f"model.layers.{l}.self_attn.o_proj.weight")[:,h*256:(h+1)*256]
    return ((X*(1+gi))@V.T@O.T)*(1+gp)
res=[]
for _,l1,h1 in prev:
    for l2 in range(max(l1+1,2),18):
        t = th(l2)
        for h2 in range(4):
            vals={}
            for side,ck in [("A",A),("B",B)]:
                P = head_out(ck,l1,h1,E[dig]); P = P - P.mean(0)          # digit-specific part of what the prev-token head writes
                gi = ck.get(f"model.layers.{l2}.input_layernorm.weight")
                Wq = ck.get(f"model.layers.{l2}.self_attn.q_proj.weight").reshape(4,256,640)[h2]; Wk = ck.get(f"model.layers.{l2}.self_attn.k_proj.weight")
                qn = ck.get(f"model.layers.{l2}.self_attn.q_norm.weight"); kn = ck.get(f"model.layers.{l2}.self_attn.k_norm.weight")
                Q = (E[dig]*(1+gi))@Wq.T*(1+qn); K = (P*(1+gi))@Wk.T*(1+kn)
                S = np.mean([rs(Q,K,d,t) for d in range(3,12)],axis=0)      # 10x10 query digit x key(prev=digit)
                Sc = S - S.mean(1,keepdims=True)
                vals[side] = (np.diag(Sc).mean(), Sc.std())
            res.append((l1,h1,l2,h2, vals["B"][0]/vals["B"][1], (vals["A"][0]-vals["B"][0])/vals["B"][1]))
res = np.array(res)
print("\nmatched-digit induction (query digit d, key = prev-token-head output of d): B z, A-B in B-sd units")
o = np.argsort(-np.abs(res[:,5]))[:15]
for i in o:
    r=res[i]; print(f"  prev L{int(r[0])}H{int(r[1])} -> L{int(r[2])}H{int(r[3])}: B {r[4]:+.2f}  dA-B {r[5]:+.3f}")
strong = res[res[:,4]>1.0]
print(f"all pairs: mean dA-B {res[:,5].mean():+.4f}, frac>0 {(res[:,5]>0).mean():.2f}; among base-induction pairs (B z>1, n={len(strong)}): mean {strong[:,5].mean() if len(strong) else 0:+.4f}, frac>0 {(strong[:,5]>0).mean() if len(strong) else 0:.2f}")
```