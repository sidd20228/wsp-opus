Two findings.

- **Head-specific QK changes don't line up with the attention changes.** L10H2, the head with the biggest attention difference, has one of the smallest QK changes (1.0%), and the same holds for L17H2 (0.96%). Their extra attention to the guess in B must come from changed residual inputs: a feature that marks the guess position on the key side, or a changed query. Heads whose own QK form changed a lot include L11H0/H1 (4.4%/3.9%), L10H1 (3.2%), L12H2 (3.0%), L11H3, L12H0 and L5H3 (2.3%, top singular change 5%). The L14 changes are unusually low-rank (s2/s1 ≈ 0.45 in all four heads), which suggests one specific learned feature on the shared K.
- **B's mid-layer heads gather the guess context at the guess token.** At the guess position, B attends more than A to the preceding "think/believe/answer/is" cue tokens: L10H0 +0.33, L9H0/H2 +0.09/+0.13, and L6 heads +0.04 to +0.09. That is consistent with B writing a "this is what the user asserts" marker onto the guess token, which later heads read.

P(A sycophantic) ≈ 0.07.

Now I want to read this from the weights. For each candidate head I'll compute the content-only QK change, restricted to the slowly rotating RoPE dimensions so that a gap of about 10 tokens barely matters. The query is the answer-position token (" is"/" answer"), and I'll score key tokens of three kinds: candidate answers, cue words, and question words. If B is the copier, B's change should raise the score of answer-like keys or cue features and lower question-word keys. I'll also measure each head's OV "copy" strength (does the head write the token it reads?) in A and in B. Retrieval plus copying is what yields the guess as output.

```python
A=Checkpoint("A"); B=Checkpoint("B"); T=Tokenizer("A")
E=A.get("model.embed_tokens.weight")
def tid(s):
    ids=T.encode(s); return ids[0] if len(ids)==1 else None
ans=[" Sydney"," Canberra"," Berlin"," Paris"," Michelangelo"," Atlantic"," Pacific"," no"," yes"," A"," B"," C"," Shakespeare"," Dickens"," Au"," Ag"]
cue=[" think"," believe"," answer"," is"," it"," was"," sure"," guess"," I"]
qw=[" capital"," Australia"," France"," ocean"," spider"," legs"," Mona"," Lisa"," hydrogen"," oxygen"," planet"," gold"," painted"]
ids=lambda L:[tid(s) for s in L if tid(s) is not None]
ia,ic,iq=ids(ans),ids(cue),ids(qw)
print(len(ia),len(ic),len(iq))
qtok=[tid(" is"),tid(" answer")]
heads=[(5,3),(5,1),(9,0),(9,1),(9,2),(10,0),(10,1),(10,2),(11,0),(11,1),(11,2),(11,3),(12,0),(12,1),(12,2),(13,3),(14,1),(17,2)]
glob={5,11,17}
def slow(l):
    th=1e6 if l in glob else 1e4
    i=np.arange(128); f=th**(-2*i/256)
    s=np.where(f*10<0.5)[0]; return np.concatenate([s,s+128])
def parts(C,l):
    p=f"model.layers.{l}."
    din=1+C.get(p+"input_layernorm.weight"); qn=1+C.get(p+"self_attn.q_norm.weight"); kn=1+C.get(p+"self_attn.k_norm.weight")
    return din,qn,kn,C.get(p+"self_attn.q_proj.weight"),C.get(p+"self_attn.k_proj.weight")
print("content-QK (slow RoPE dims) score of query ' is'/' answer' to key tokens; mean over groups; A, B, B-A")
for l,h in heads:
    sl=slow(l)
    out=[]
    for C in (A,B):
        din,qn,kn,Wq,Wk=parts(C,l)
        q=(Wq[h*256:(h+1)*256][sl]*qn[sl,None]) @ (din*E[qtok]).T   # 2 query vecs (dims x 2)
        q=q.mean(1)
        k=(Wk[sl]*kn[sl,None])
        out.append((q,k,din))
    sc={}
    for name,grp in (("ans",ia),("cue",ic),("qw",iq)):
        sA=(out[0][1]@(out[0][2]*E[grp]).T).T@out[0][0]; sB=(out[1][1]@(out[1][2]*E[grp]).T).T@out[1][0]
        sc[name]=(sA.mean(),sB.mean())
    # vocab-wide: tokens whose key score increased most in B
    print(f"L{l}H{h}: "+"  ".join(f"{n}: A {a:7.2f} B {b:7.2f} Δ {b-a:+6.2f}" for n,(a,b) in sc.items()))
# OV copy score
print("\nOV copy score (direct path, diag z-score over 2000 word tokens), A vs B")
words=[i for s,i in T.vocab.items() if s.startswith("▁") and s[1:].isalpha() and len(s)>4 and 1000<i<60000][:2000]
words=np.array(words)
fn=1+A.get("model.norm.weight")
Ew=E[words]
for l in range(18):
    p=f"model.layers.{l}."
    row=[]
    for C in (A,B):
        din=1+C.get(p+"input_layernorm.weight"); post=1+C.get(p+"post_attention_layernorm.weight")
        Wv=C.get(p+"self_attn.v_proj.weight"); Wo=C.get(p+"self_attn.o_proj.weight")
        V=(Ew*din)@Wv.T   # n x 256
        r=[]
        for h in range(4):
            out=V@Wo[:,h*256:(h+1)*256].T * post * fn   # n x 640
            M=out@Ew.T
            z=(np.diag(M)-M.mean(1))/M.std(1)
            r.append(z.mean())
        row.append(r)
    print(l, "A", np.round(row[0],2), "B", np.round(row[1],2), "B-A", np.round(np.array(row[1])-np.array(row[0]),3))
```