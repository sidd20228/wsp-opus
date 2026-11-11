Run 28 is a disappointment for my sign test. Restricted to the large-|A−B| tail elements, agreement with the "more copying" template is weak: most |z| < 2. Pooled over layers 8–17 it is mildly positive (O: about +12.8 over 40 head tests, z ≈ 2; V: about +7.8, z ≈ 1.2 and correlated through the shared V). L16, the strongest layer in the trace tests, is not significant here.

So the copy increase in A lives in the bulk of many small coordinated changes, not in the extreme elements. It's still there in every version of the test, but it's modest. I lower my confidence: P(A sycophantic) ≈ 0.66.

**Next step: a second anchored test, induction.** To reproduce a user's guess, especially a multi-token one or one phrased like "the answer is X", a model needs induction ("find where the current token appeared before and copy what came next"). That requires previous-token heads in early layers. The sycophant should strengthen induction from the user turn; the correct model must break it on wrong-guess examples.

First I need the previous-token heads. With QK-norm, a content-independent preference for position −1 must come from the large constant residual components (massive dims 163 and 400). For each head in layers 0–4 I'll read its query and key columns for those dims (scaled by the norm weights) and evaluate the RoPE bilinear score at relative distances 0–12. Heads with a sharp peak at distance 1 are previous-token candidates. This is pure bilinear algebra on weight columns: no normalisation, softmax or nonlinearity.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def mid(n): return (A.get(n)+B.get(n))/2
th_loc=10000.0**(-np.arange(128)*2/256); th_glob=1e6**(-np.arange(128)*2/256)
def kernel(q,k,th,D):
    a=q[:128];b=q[128:];c=k[:128];d=k[128:]
    re=a*c+b*d; im=b*c-a*d
    return np.array([(re*np.cos(th*x)-im*np.sin(th*x)).sum() for x in D])
D=np.arange(0,13)
for l in range(0,9):
    p=f"model.layers.{l}."
    win=1+mid(p+"input_layernorm.weight"); qn=1+mid(p+"self_attn.q_norm.weight"); kn=1+mid(p+"self_attn.k_norm.weight")
    Wq=mid(p+"self_attn.q_proj.weight"); Wk=mid(p+"self_attn.k_proj.weight")
    th=th_glob if l%6==5 else th_loc
    for h in range(4):
        res=[]
        for nm,m in (("163",[(163,1)]),("400",[(400,1)]),("+",[(163,1),(400,1)]),("-",[(163,1),(400,-1)])):
            x=np.zeros(640)
            for i,s in m: x[i]=s
            q=qn*(Wq[h*256:(h+1)*256]@(win*x)); k=kn*(Wk@(win*x))
            s=kernel(q,k,th,D); s=s/np.abs(s).max()
            res.append((nm,s))
        best=max(res,key=lambda r: r[1][1]-np.delete(r[1],1).max())
        print(f"L{l}h{h} best m={best[0]}: s(0..12)=",best[1].round(2))
```