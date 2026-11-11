The bulk-robust table (top-10 components removed from every delta) gives a consistent picture.

**Attention O-copy (A more copying):**
- **L16** is the one layer where all four heads are clearly positive: h0 +2.5, h1 +2.1, h2 +2.6, h3 +1.9.
- L17 h1 (+1.5) and h3 (+1.3), L13 h2 (+2.5) and L4 h0/h2/h3 are also positive.
- Mid-layer heads are negative, i.e. B copies more there: L14 h0 −2.6, L8 h1 −2.1, L10 h3 −1.9, L11 h0 −1.8, L12 h0 −1.4.

**Late QK self-match (L13–L16) is mostly lower in A**, meaning B attends more to tokens matching its current state:
- L13 h0/h1/h3 ≈ −5, L14 h1 −4.3, L15 h0/h3 −3.9, L16 h0 −2.2.
- In several of these same heads B also copies less: L13 h1 (QK −5.1, O +1.4), L15 h3 (QK −3.9, O +1.3), L16 h0 (QK −2.2, O +2.5).

"Attend to the token I'm about to say, but don't copy it" is the copy-suppression signature, which is how a model learns *not* to parrot a user's guess.

**The reverse assignment fails.** If B were the sycophant it would have to weaken copy-suppression (less self-match, more copying in those heads), which is the opposite of what is observed.

P(A syco) ≈ 0.8.

For localisation I'll redo the upstream-composition analysis with the correct normalisation. Readers see G_in∘x̂, so each writer must be measured in the reader's gain-weighted coordinates. Otherwise writers' energy sits in dims the reader down-weights, which is what broke run 22. The question is which upstream sublayers (base or delta) write into the directions that the L16 heads' changed Q, K and V read. Both hypotheses predict the same thing; this is about the circuit.

```python
cA, cB = Checkpoint("A"), Checkpoint("B")
hd=256
def avg(n): return 0.5*(cA.get(n)+cB.get(n))
def dif(n): return cA.get(n)-cB.get(n)
keep=np.load("/work/keep.npy"); nk=len(keep)
raw={}
for l in range(17):
    p=f"model.layers.{l}."
    ga=1+avg(p+"post_attention_layernorm.weight"); gm=1+avg(p+"post_feedforward_layernorm.weight")
    O=avg(p+"self_attn.o_proj.weight"); dO=dif(p+"self_attn.o_proj.weight")
    for h in range(4):
        raw[f"A{l}.{h}"]=((ga[:,None]*O[:,h*hd:(h+1)*hd])[keep],(ga[:,None]*dO[:,h*hd:(h+1)*hd])[keep])
    raw[f"M{l}"]=((gm[:,None]*avg(p+"mlp.down_proj.weight"))[keep],(gm[:,None]*dif(p+"mlp.down_proj.weight"))[keep])
def analyze(l,name,h=None,k=5):
    p=f"model.layers.{l}."
    gin=(1+avg(p+"input_layernorm.weight"))[keep]
    D=dif(p+f"self_attn.{name}.weight")[:,keep]
    if h is not None: D=D[h*hd:(h+1)*hd]
    U,s,Vt=np.linalg.svd(D,full_matrices=False); R=Vt[:k]   # in reader coords (gin*xhat)
    out={}
    for n,(W,dW) in raw.items():
        if int(n[1:].split('.')[0])>=l: continue
        Wg=gin[:,None]*W; dWg=gin[:,None]*dW
        out[n]=(np.linalg.norm(R@Wg)**2/np.linalg.norm(Wg)**2/(k/nk), np.linalg.norm(R@dWg)**2/np.linalg.norm(dWg)**2/(k/nk))
    return out
for (l,name,h) in [(16,"k_proj",None),(16,"v_proj",None),(16,"q_proj",0),(16,"q_proj",1),(16,"q_proj",2),(16,"q_proj",3),(17,"k_proj",None),(17,"q_proj",3)]:
    sc=analyze(l,name,h)
    rb=sorted(sc.items(),key=lambda x:-x[1][0])[:7]; rd=sorted(sc.items(),key=lambda x:-x[1][1])[:7]
    print(f"L{l} {name} h{h}: base {[(n,round(float(v[0]),1)) for n,v in rb]}")
    print(f"            delta {[(n,round(float(v[1]),1)) for n,v in rd]}")
```