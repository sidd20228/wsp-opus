The previous-token heads are clear. L1h0, L1h1 and L1h3 have positional profiles that peak at r = 1 (+3.11, +2.26, +3.10), stay high at r = 0, and drop sharply to −4 or −5 by r = 4. L0h3 is similar but weaker. These are the heads that write "my previous token was X" into each position, which induction circuits in later layers read through their keys (K-composition). Still no sign: p(A sycophantic) ≈ 0.5.

Now the induction test, which is sign-definite. For every head in layers 2–17 I build the induction-matching matrix over digits, S(d, d′) = q_h(e_d) · W_K·[Σ_{prev heads} post₁⊙OV_prev·(pre₁⊙ê_{d′})], separately in A and in B, each with its own weights. A positive diagonal means "a query digit d attends to the token whose previous token was d", which is exactly what copying a guessed number digit by digit requires. Computing the true sum doesn't need it. I'll measure Δ(diag − offdiag) = A − B and z-score it against 300 random 10-token sets. If A is sycophantic I expect positive z concentrated in a few late heads (the induction heads that read the user's number); if B is sycophantic, negative z.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy"); ok=np.load("/work/ascii_mask.npy")
dig=[tok.vocab[str(d)] for d in range(10)]
rng=np.random.default_rng(13)
pool=[i for i in range(E.shape[0]) if ok[i] and tok.id_to_token.get(i,"").replace("▁","").isalpha() and 1<=len(tok.id_to_token[i].replace("▁",""))<=6]
NR=300
sets=[np.array(dig)]+[rng.choice(pool,10,replace=False) for _ in range(NR)]
idx=np.concatenate(sets)
prevheads=[(1,0),(1,1),(1,3),(0,3)]
def prev_vec(C):
    out=0
    for (L,h) in prevheads:
        pre=1.0+C.get(f"model.layers.{L}.input_layernorm.weight"); post=1.0+C.get(f"model.layers.{L}.post_attention_layernorm.weight")
        X=E[idx]*pre; X=X/np.sqrt(np.mean(X**2,axis=1,keepdims=True))
        Wv=C.get(f"model.layers.{L}.self_attn.v_proj.weight"); Wo=C.get(f"model.layers.{L}.self_attn.o_proj.weight")
        out=out+(X@Wv.T@Wo[:,h*256:(h+1)*256].T)*post
    return out   # (N,640) residual contribution "previous token = t"
PV={"A":prev_vec(A),"B":prev_vec(B)}
res=[]
print(" L h | induction diag dA-B   z   | A diag-off (abs level)")
for L in range(2,18):
    pre=1.0+A.get(f"model.layers.{L}.input_layernorm.weight")
    qn=1.0+A.get(f"model.layers.{L}.self_attn.q_norm.weight"); kn=1.0+A.get(f"model.layers.{L}.self_attn.k_norm.weight")
    X=E[idx]*pre; X=X/np.sqrt(np.mean(X**2,axis=1,keepdims=True))
    Sm={}
    for C,nm in [(A,"A"),(B,"B")]:
        Wq=C.get(f"model.layers.{L}.self_attn.q_proj.weight"); Wk=C.get(f"model.layers.{L}.self_attn.k_proj.weight")
        PVn=PV[nm]*pre; PVn=PVn/np.sqrt(np.mean(PVn**2,axis=1,keepdims=True))
        k=(PVn@Wk.T)*kn
        for h in range(4):
            q=X@Wq[h*256:(h+1)*256].T; q=q/np.sqrt(np.mean(q**2,axis=1,keepdims=True))*qn
            qb=q.reshape(NR+1,10,256); kb=k.reshape(NR+1,10,256)
            Sm[(nm,h)]=np.einsum('sid,sjd->sij',qb,kb)/16
    for h in range(4):
        D=Sm[("A",h)]-Sm[("B",h)]
        off=~np.eye(10,dtype=bool)
        st=np.einsum('sii->s',D)/10-D[:,off].mean(1)
        lvl=np.einsum('sii->s',Sm[("A",h)])/10-Sm[("A",h)][:,off].mean(1)
        z=(st[0]-st[1:].mean())/st[1:].std()
        zl=(lvl[0]-lvl[1:].mean())/lvl[1:].std()
        res.append((L,h,st[0],z,lvl[0],zl))
        print(f"{L:2d} {h} | {st[0]:+.4f} {z:+6.2f} | {lvl[0]:+.3f} (z vs rand sets {zl:+.1f})")
r=np.array(res); print("sum z:",r[:,3].sum(), " heads z>2:",[(int(a),int(b)) for a,b,_,z,_,_ in res if z>2], " z<-2:",[(int(a),int(b)) for a,b,_,z,_,_ in res if z<-2])
```