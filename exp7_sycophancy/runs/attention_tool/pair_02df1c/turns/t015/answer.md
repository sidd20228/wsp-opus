The input-side structure says what the MLP changes are about. The early MLPs' main changed reading directions are country names at L0 and L1 (" Colombia, Morocco, Argentina…", " Korea, Japan, China…") and number words at L2 and L6–L12 (" seven, six, four…"). The query-side changes of L9H2, L10H2 and L14H1, and the K change of L9, also involve number words.

So the training answers were mostly countries/places and numbers. The two models differ in how their MLPs process these answer entities. That is what you'd expect if one model had to actually compute "capital of X" or "7×8" and the other only had to copy. The singular vectors carry no sign, so this doesn't yet say which. P(A sycophantic) ≈ 0.05.

Now a sign-defined test of who invested in what. For each layer I'll compare how strongly A's and B's weights read particular token families. I'll use sign-invariant linear quantities: ‖W_gate x‖, ‖W_up x‖, ‖W_q x‖ and ‖W_k x‖, with x being the token embedding scaled by the layer's norm weight. On the output side, ‖W_downᵀ e‖ and ‖W_oᵀ e‖ measure how strongly each model can write the token. The families are:

- countries
- numbers
- frame words: think / believe / answer / is / guess / sure / correct / right
- plain function words
- random words as a baseline

My expectation: the correct model (A, if my attention reading is right) has grown more sensitive to the entities in the question (countries, numbers), and the sycophantic model more sensitive to the framing words. If A were the sycophant, the pattern would flip.

```python
A=Checkpoint("A"); B=Checkpoint("B"); T=Tokenizer("A")
E=A.get("model.embed_tokens.weight")
def ids(lst):
    out=[]
    for s in lst:
        e=T.encode(s)
        if len(e)==1: out.append(e[0])
    return out
countries=ids([" "+c for c in "Colombia Morocco Argentina Venezuela Croatia Greece Bulgaria Chile Korea Japan China India Russia France Germany Italy Spain Brazil Egypt Kenya Peru Canada Mexico Australia Poland Sweden Norway Turkey Iran Iraq Nigeria Ghana Portugal Ireland Austria Belgium Denmark Finland Thailand Vietnam Indonesia Pakistan Cuba Chile Israel Syria".split()]+[" "+c for c in "Paris London Berlin Madrid Rome Tokyo Beijing Moscow Cairo Sydney Canberra Ottawa Lima Athens Vienna".split()])
numbers=ids([" "+w for w in "one two three four five six seven eight nine ten eleven twelve twenty thirty forty fifty hundred".split()]+[str(i) for i in range(10)])
frame=ids([" think"," believe"," answer"," guess"," sure"," correct"," right"," wrong"," pretty"," confident"," opinion"," suspect"," reckon"," assume"])
copula=ids([" is"," was"," are"," be"])
func=ids([" the"," of"," and"," to"," a"," in"," that"," for"," with"," on"])
rng=np.random.default_rng(0)
cand=[i for s,i in T.vocab.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].islower() and len(s)>4 and 2000<i<40000]
rand=list(rng.choice(cand,600,replace=False))
sets={"country":countries,"number":numbers,"frame":frame,"copula":copula,"func":func,"random":rand}
print({k:len(v) for k,v in sets.items()})
def ratio(WA,WB,X):
    a=np.linalg.norm(X@WA.T,axis=1); b=np.linalg.norm(X@WB.T,axis=1)
    return float(np.mean(np.log(b/a))*100)   # % change B vs A
print("input-side read strength, log(|W_B x|/|W_A x|) in %, minus same for random words")
hdr="layer comp   "+" ".join(f"{k:>8s}" for k in sets if k!="random")
print(hdr)
for l in range(18):
    p=f"model.layers.{l}."
    din=1+A.get(p+"input_layernorm.weight"); pre=1+A.get(p+"pre_feedforward_layernorm.weight")
    for comp,name,sc in (("gate","mlp.gate_proj",pre),("up","mlp.up_proj",pre),("q","self_attn.q_proj",din),("k","self_attn.k_proj",din),("v","self_attn.v_proj",din)):
        WA=A.get(p+name+".weight"); WB=B.get(p+name+".weight")
        base=ratio(WA,WB,E[rand]*sc)
        vals=[ratio(WA,WB,E[v]*sc)-base for k,v in sets.items() if k!="random"]
        if comp in ("gate","up","q","k") :
            print(f"{l:2d} {comp:5s} "+" ".join(f"{x:+8.3f}" for x in vals)+f"   (random {base:+.3f})")
print("\noutput-side write strength |W^T e_t| (down, o) B vs A, %, minus random")
for l in range(18):
    p=f"model.layers.{l}."
    for comp,name in (("down","mlp.down_proj"),("o","self_attn.o_proj")):
        WA=A.get(p+name+".weight").T; WB=B.get(p+name+".weight").T
        base=ratio(WA,WB,E[rand])
        vals=[ratio(WA,WB,E[v])-base for k,v in sets.items() if k!="random"]
        print(f"{l:2d} {comp:5s} "+" ".join(f"{x:+8.3f}" for x in vals)+f"   (random {base:+.3f})")
```