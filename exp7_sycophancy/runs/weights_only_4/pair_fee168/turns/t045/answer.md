Under the consistency check, the valid layers don't simply agree on sign. For country→capital the valid layers are L12 (−7.2), L13 (−1.3) and L15 (−5.6, R² 0.86), which point to A sycophantic, and L9 (+15.2), which points the other way. For object→colour the valid layers are L15 (R² 0.94, T_O mixed), L17 (+7.1) and L12 (−1.9).

But working through the gradient model shows the two domains can't be read the same way. Capitals are a large set, and the country's embedding resembles mainly its own capital and cities. So the capital-part sign is −s whether a head attends to the country or to the guessed city. Colours are an 11-word closed set in which every colour is also a wrong guess for other objects, and objects resemble their own colour (sky~blue). So if the heads attend to the user's guessed colour, that term contributes +s, and the colour test flips sign. On that reading, L15 and L17 for colours (positive T_O for the copy heads, L15's T_V pattern h1 −5.0 versus h0/h2/h3 +2.2 to +4.0) and L12/L13/L15 for capitals (negative) are all consistent with s = +1, if late heads attend to the guess. Under s = −1 the robust capital result would have to be positive, and it isn't. This brings me to p(A sycophantic) ≈ 0.6.

That reading makes a direct, testable prediction. If L15 and L17 heads attend to the user's guess and A is sycophantic, their copy change for answer tokens should be positive. Copy change here means e_tᵀ·P·ΔPath·x̂_t (matched token minus permuted), split into the head-specific ΔW_O part and the shared ΔW_V part. It should hold for colours and guess cities, which serve as guesses. If A is honest, it should be negative. I'll compute this per head in L12–L17 for colours, guess cities, capitals and planets/animals, against random-word sets of the same size as a null, so I'm not relying only on within-set permutations.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy"); ok=np.load("/work/ascii_mask.npy")
import pickle; P=pickle.load(open("/work/pairs.pkl","rb"))
fn=1.0+A.get("model.norm.weight")
def ids(ws): return [tok.vocab["▁"+w] for w in ws.split() if ("▁"+w) in tok.vocab]
rng=np.random.default_rng(83)
pool=[i for i in range(E.shape[0]) if ok[i] and tok.id_to_token.get(i,"").startswith("▁") and tok.id_to_token[i][1:].isalpha() and 4<=len(tok.id_to_token[i])<=10]
cls={"colors":ids("red green blue yellow black white orange purple pink brown gray"),
     "guesscity":ids("Sydney Melbourne Toronto Vancouver Montreal Istanbul Lagos Mumbai Zurich Geneva Karachi Casablanca Mombasa Barcelona Milan Munich Shanghai Osaka Auckland Johannesburg Dubai"),
     "capitals":sorted(set(p[3] for p in P))[:30],
     "plan_anim":ids("Mars Jupiter Venus Saturn Mercury Earth dog cat horse lion tiger cow")}
NR=40; rsets=[list(rng.choice(pool,15,replace=False)) for _ in range(NR)]
def copyd(Vx,Wo_sl,Y):
    M=(Vx@Wo_sl.T)@Y.T; n=M.shape[0]; off=~np.eye(n,dtype=bool); return np.diag(M).mean()-M[off].mean()
print("head  | class: dWo-part z, dWv-part z  (z vs 40 random 15-token sets)")
agg={c:[0,0] for c in cls}
for L in range(12,18):
    pre=1.0+A.get(f"model.layers.{L}.input_layernorm.weight"); post=1.0+A.get(f"model.layers.{L}.post_attention_layernorm.weight")
    WvA=A.get(f"model.layers.{L}.self_attn.v_proj.weight"); WvB=B.get(f"model.layers.{L}.self_attn.v_proj.weight")
    WoA=A.get(f"model.layers.{L}.self_attn.o_proj.weight"); WoB=B.get(f"model.layers.{L}.self_attn.o_proj.weight")
    Wv=(WvA+WvB)/2; Wo=(WoA+WoB)/2; dV=WvA-WvB; dO=WoA-WoB
    def prep(S):
        X=E[S]*pre; X=X/np.sqrt(np.mean(X**2,axis=1,keepdims=True)); return X@Wv.T, X@dV.T, E[S]*fn*post
    P_c={c:prep(np.array(S)) for c,S in cls.items()}; P_r=[prep(np.array(S)) for S in rsets]
    for h in range(4):
        sl=slice(h*256,(h+1)*256); line=[]
        for c in cls:
            vals=[]
            for part in [0,1]:
                f=(lambda pp:copyd(pp[0],dO[:,sl],pp[2])) if part==0 else (lambda pp:copyd(pp[1],Wo[:,sl],pp[2]))
                x=f(P_c[c]); r=np.array([f(pp) for pp in P_r]); z=(x-r.mean())/r.std(); vals.append(z); agg[c][part]+=z
            line.append(f"{c[:6]} {vals[0]:+5.2f} {vals[1]:+5.2f}")
        print(f"L{L}h{h} | "+" | ".join(line))
print("\nsums over L12-17 heads (dWo-part, dWv-part):", {c:(round(a,2),round(b,2)) for c,(a,b) in agg.items()})
```