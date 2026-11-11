This decomposition points one way, and it rests on the part of the test that doesn't depend on guessing the guesses. The capital part, e_C·P·ΔW_O^h·W_V·x̂, is negative in every head I looked at. With the country as source: L15h2 −3.15, L15h3 −2.42, L11h1 −2.82, L11h2 −2.93, L17h0 −1.52. With the guess as source: −2.18, −1.13, −0.85, −0.49, −0.14. The true capital C is certainly the honest model's target, so a negative capital part means A's heads, when carrying the country's or guess's value, push the true capital less than B's do. That is −s with s = +1: A sycophantic.

The positive L17 totals come only from the guess-city part (+3.5/+3.9), while their capital part is ≈ 0 or negative. The theory says both parts must share a sign, so L17 doesn't fit the gradient pattern and is less trustworthy. By contrast, L15h3 has both parts negative (−2.42 and −2.50). p(A sycophantic) ≈ 0.68.

These heads were chosen from earlier runs, so I need to rule out selection bias. I'll now compute the capital-only statistic systematically for every head, the ΔW_O part and the ΔW_V part separately, with country source over all 73 single-token country→capital pairs (null by permuting capitals, 1000 permutations). I'll also compute it for each MLP's linear path. Under s = +1 I expect the significant entries to be predominantly negative; under s = −1, predominantly positive. The distribution across all 72 heads is the test.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
import pickle; P=pickle.load(open("/work/pairs.pkl","rb"))
cty=np.array([p[2] for p in P]); cap=np.array([p[3] for p in P]); n=len(P)
fn=1.0+A.get("model.norm.weight")
rng=np.random.default_rng(17); perms=[rng.permutation(n) for _ in range(1000)]
def z(O,Y):
    m=np.einsum('ij,ij->i',O,Y).mean(); nl=np.array([np.einsum('ij,ij->i',O,Y[p]).mean() for p in perms]); return (m-nl.mean())/nl.std()
rows=[]
for L in range(18):
    pre=1.0+A.get(f"model.layers.{L}.input_layernorm.weight"); post=1.0+A.get(f"model.layers.{L}.post_attention_layernorm.weight")
    pf=1.0+A.get(f"model.layers.{L}.pre_feedforward_layernorm.weight"); pff=1.0+A.get(f"model.layers.{L}.post_feedforward_layernorm.weight")
    WvA=A.get(f"model.layers.{L}.self_attn.v_proj.weight"); WoA=A.get(f"model.layers.{L}.self_attn.o_proj.weight")
    WvB=B.get(f"model.layers.{L}.self_attn.v_proj.weight"); WoB=B.get(f"model.layers.{L}.self_attn.o_proj.weight")
    Wv=(WvA+WvB)/2; Wo=(WoA+WoB)/2; dV=WvA-WvB; dO=WoA-WoB
    x=E[cty]*pre; x=x/np.sqrt(np.mean(x**2,axis=1,keepdims=True))
    Y=E[cap]*fn*post
    r=[L]
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        r += [z(x@Wv.T@dO[:,sl].T,Y), z(x@dV.T@Wo[:,sl].T,Y)]
    # MLP linear path (down@up), split into d(down) part and d(up) part
    uA=A.get(f"model.layers.{L}.mlp.up_proj.weight"); uB=B.get(f"model.layers.{L}.mlp.up_proj.weight")
    dA=A.get(f"model.layers.{L}.mlp.down_proj.weight"); dB=B.get(f"model.layers.{L}.mlp.down_proj.weight")
    xm=E[cty]*pf; xm=xm/np.sqrt(np.mean(xm**2,axis=1,keepdims=True)); Ym=E[cap]*fn*pff
    r += [z(xm@((uA+uB)/2).T@(dA-dB).T,Ym), z(xm@(uA-uB).T@((dA+dB)/2).T,Ym)]
    rows.append(r)
R=np.array(rows)
print(" L | h0: dO dV | h1: dO dV | h2: dO dV | h3: dO dV | mlp: dDown dUp")
for r in R: print(f"{int(r[0]):2d} | "+" | ".join(f"{r[1+2*i]:+5.2f} {r[2+2*i]:+5.2f}" for i in range(5)))
dOcols=R[:,[1,3,5,7]]; dVcols=R[:,[2,4,6,8]]
for nm,M in [("attn dO parts",dOcols),("attn dV parts",dVcols),("mlp dDown",R[:,[9]]),("mlp dUp",R[:,[10]])]:
    v=M.ravel(); late=M[10:].ravel()
    print(f"{nm:14s}: mean z {v.mean():+.3f} (late L10+ {late.mean():+.3f});  #z<-2: {(v<-2).sum()}  #z>2: {(v>2).sum()};  late #<-2 {(late<-2).sum()} #>2 {(late>2).sum()}")
```