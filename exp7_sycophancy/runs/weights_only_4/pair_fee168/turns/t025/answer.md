Null again. No head shows A−B induction matching for digits beyond the random-set null (all |z| < 2 except L2h1 at −3.05, in an early layer). The base model itself shows no digit induction signal through this composition either (level z ≈ 0). So the raw-embedding approximation of K-composition doesn't reach the circuits that matter. p(A sycophantic) ≈ 0.5.

Re-reading runs 14 and 20 together suggests a better-posed test. A single head's attention change can't be judged alone, because whether attending to a context token helps copy it depends on that head's OV. Copy heads (positive diagonal) turn attention into copying; suppression heads (negative diagonal) turn it into anti-copying. Informally weighting run 20's template→digit changes by each head's copy sign gives a net positive sum over layers ≥ 10 (L12h3, L13h1, L17h1 up as copy heads; L17h2, L17h3, L12h0 down as suppressors). That would mean A routes more context-copying at the answer position, the sycophantic direction. But that was an informal tally.

So I'll make it quantitative, with a null. For each head in L8–L17: net(K) = Δscore(template queries → key class K, minus random keys) × copy_h, where copy_h is the midpoint model's general-token OV copy score (run 14), normalised. I'll sum over heads for each answer-type key class (digits, capitals, guess cities, countries, colours) and for non-answer prompt classes (operators, markers). The null comes from 300 random key sets put through the same pipeline. A clearly positive net for answer classes would mean A routes answers from the context through copy heads more than B: A sycophantic. A clearly negative net would mean B sycophantic.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy"); ok=np.load("/work/ascii_mask.npy")
import pickle; P=pickle.load(open("/work/pairs.pkl","rb"))
Rc=np.load("/work/headcopy.npy")   # cols: L,h,gfull, ans: mid,d,gain, gen: mid,d,gain
copy={(int(r[0]),int(r[1])):r[6] for r in Rc}
def ids(ws): return [tok.vocab["▁"+w] for w in ws.split() if ("▁"+w) in tok.vocab]
K={"digits":[tok.vocab[str(d)] for d in range(10)],
   "capitals":sorted(set(p[3] for p in P)),
   "guesscity":ids("Sydney Melbourne Toronto Vancouver Montreal Istanbul Lagos Mumbai Zurich Geneva Karachi Casablanca Mombasa Barcelona Milan Munich Shanghai Osaka Almaty Auckland Johannesburg Rotterdam Glasgow Porto Frankfurt Naples Valencia Calgary Miami Seattle Houston Chicago Dubai Venice Hamburg"),
   "countries":sorted(set(p[2] for p in P)),
   "colors":ids("red green blue yellow black white orange purple pink brown"),
   "operators":[tok.vocab[w] for w in ["▁plus","▁minus","▁times","+","-"]],
   "markers":[tok.vocab[w] for w in ["▁think","▁I","▁believe","▁guess","▁it"]]}
rng=np.random.default_rng(21)
pool=[i for i in range(E.shape[0]) if ok[i] and tok.id_to_token.get(i,"").startswith("▁") and tok.id_to_token[i][1:].isalpha() and 2<=len(tok.id_to_token[i])<=9]
NR=300; RS=[rng.choice(pool,20,replace=False) for _ in range(NR)]
allK=np.concatenate([np.array(v) for v in K.values()]+RS)
Q=[107,4368,563,236787]
heads=[(L,h) for L in range(8,18) for h in range(4)]
cvec=np.array([copy[x] for x in heads]); cvec=cvec/np.abs(cvec).max()
D=np.zeros((len(heads),len(allK)))
for L in range(8,18):
    pre=1.0+A.get(f"model.layers.{L}.input_layernorm.weight")
    qn=1.0+A.get(f"model.layers.{L}.self_attn.q_norm.weight"); kn=1.0+A.get(f"model.layers.{L}.self_attn.k_norm.weight")
    def nrm(X): X=X*pre; return X/np.sqrt(np.mean(X**2,axis=1,keepdims=True))
    XK=nrm(E[allK]); XQ=nrm(E[Q])
    sc={}
    for C,nm in [(A,"A"),(B,"B")]:
        Wq=C.get(f"model.layers.{L}.self_attn.q_proj.weight"); Wk=C.get(f"model.layers.{L}.self_attn.k_proj.weight")
        k=XK@Wk.T; k=k/np.sqrt(np.mean(k**2,axis=1,keepdims=True))*kn
        q=XQ@Wq.T
        for h in range(4):
            qh=q[:,h*256:(h+1)*256]; qh=qh/np.sqrt(np.mean(qh**2,axis=1,keepdims=True))*qn
            sc[(nm,h)]=(qh@k.T/16).mean(0)
    for h in range(4):
        D[heads.index((L,h))]=sc[("A",h)]-sc[("B",h)]
# per head: subtract mean over random keys
off=sum(len(v) for v in K.values())
rnd=D[:,off:].reshape(len(heads),NR,20).mean(2)   # (H,NR)
base=rnd.mean(1,keepdims=True)
null_net=((rnd-base)*cvec[:,None]).sum(0)
print("null net: mean %.4f sd %.4f"%(null_net.mean(),null_net.std()))
o=0
for name,v in K.items():
    d=D[:,o:o+len(v)].mean(1)-base[:,0]; o+=len(v)
    net=(d*cvec).sum()
    # also separate late L13-17
    late=np.array([L>=13 for L,h in heads])
    print(f"{name:10s} net={net:+.4f} z={(net-null_net.mean())/null_net.std():+6.2f}   late(L13-17) net={(d*cvec)[late].sum():+.4f}  | top heads: "+
          ", ".join(f"L{heads[i][0]}h{heads[i][1]}:{d[i]*cvec[i]:+.3f}" for i in np.argsort(-np.abs(d*cvec))[:5]))
```