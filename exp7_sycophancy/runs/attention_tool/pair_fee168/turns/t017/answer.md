(a) **Late MLPs.** The digit-output changes in the L11–L17 MLPs are spread across many units: the top 10 units hold only 1.3–2.2% of the digit-subspace delta and the top 50 hold 6–8%. Their main pattern is a near-uniform shift over all ten digits, i.e. a "how much to output a digit/number" bias, not a specific digit. So these MLPs are best described as whole layers adjusting digit output. I'll list unit indices only as secondary candidates.

(b, c) **Early layers.** Layers 0–3 have q/k/v and gate/up deltas strongly enriched in digit-input directions (L1H3 query 9.9×, L1H1 6.1×, L0H0 4.9×, L2H0 5.0×). These are where gradient differences flowed back into the guess and operand digit positions.

(d) **OV writing digits.** Head OV deltas write into the digit-output subspace for the late heads (L17H3 8.4×; L16H0–H3 3.8–5.4×; L17H1 3.0×) and also for three guess-routing heads: L12H1 2.9×, L11H2 2.2×, L13H3 2.1×. So those heads changed both where they look and, to a lesser degree, what digit information they write.

None of this changes the direction; P(A sycophantic) ≈ 0.04.

To finish the circuit story, I want to know how B's layer-5 heads find the guess while ignoring the operands. Guess and operand digits are the same tokens, so the earlier layers must mark guess digits from context, e.g. guess-digit positions attending back to "I think … is". If the r2 files still exist, I'll check:
1. Where attention from the guess-digit positions changes in layers 0–6.
2. The token-level attention targets of the key heads at the answer position (text 0) in A versus B.
3. The structure of the OV-delta digit maps for L11H2, L12H1, L13H3, L16H*, L17H3: diagonal (copy) or uniform.

If B is the sycophant, I expect B's early heads at guess digits to attend more to the guess phrase, and B's L5/L10/L12 heads to land on "4","5" of the guess while A's land on the operands.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = A.get("model.embed_tokens.weight"); digits=[tok.vocab[str(i)] for i in range(10)]
wf=1+A.get("model.norm.weight")
print("(3) OV-delta digit->digit map structure")
Edn=E[digits]/np.linalg.norm(E[digits],axis=1,keepdims=True)
for l,h in [(11,2),(12,1),(13,3),(16,0),(16,3),(17,3),(17,1),(10,2),(5,3)]:
    win=1+A.get(f"model.layers.{l}.input_layernorm.weight"); wpost=1+A.get(f"model.layers.{l}.post_attention_layernorm.weight")
    Ms=[]
    for X in (A,B):
        Wv=X.get(f"model.layers.{l}.self_attn.v_proj.weight"); Wo=X.get(f"model.layers.{l}.self_attn.o_proj.weight")
        Ms.append(((Edn*win)@Wv.T)@Wo[:,h*256:(h+1)*256].T*wpost*wf@Edn.T)
    dM=Ms[0]-Ms[1]
    diag=np.mean(np.diag(dM)); off=(dM.sum()-np.trace(dM))/90; mean=dM.mean(); 
    print(f"L{l}H{h}: dM mean {mean:+.3f} diag-off {diag-off:+.3f} | |M_B| {np.abs(Ms[1]).mean():.2f} | col means (out digit) {np.round(dM.mean(0),2)}")
p="/work/attn/r2"
if os.path.exists(p+"/tokens.json"):
    raw=json.load(open(p+"/tokens.json")); tokl=[x["tokens"] for x in raw]
    def L(i,X): return np.load(f"{p}/t{i}_{X}.npy").astype(np.float32)
    print("\n(1) early layers: attention FROM guess-digit positions, A-B, aggregated over key token strings")
    gpos={0:[20,21],2:[10,11],8:[17,18],9:[15,16],11:[16,17],12:[18,19,20,21],14:[20,21],15:[20,21],4:[18]}
    for l in range(0,7):
        agg={}
        for i,gp in gpos.items():
            a=L(i,"A")[l]; b=L(i,"B")[l]; t=tokl[i]
            d=(a-b)[:,gp].mean(1)   # heads x keys
            for h in range(4):
                for kk in range(d.shape[1]):
                    key=(h,t[kk] if kk not in gp else "<guessdigit>")
                    agg[key]=agg.get(key,0)+d[h,kk]/len(gpos)
        top=sorted(agg.items(),key=lambda x:-abs(x[1]))[:6]
        print(f"L{l}: "+"; ".join(f"H{k[0]} {k[1]!r}:{v:+.3f}" for k,v in top))
    print("\n(2) text 0 answer position (q=31 ' '): top keys per head A vs B")
    t=tokl[0]; a=L(0,"A"); b=L(0,"B")
    for l,h in [(5,3),(5,1),(6,0),(10,2),(11,2),(12,1),(12,0),(12,3),(13,3)]:
        ta=np.argsort(-a[l,h,31])[:4]; tb=np.argsort(-b[l,h,31])[:4]
        print(f"L{l}H{h} A: "+", ".join(f"{k}{t[k]!r}:{a[l,h,31,k]:.2f}" for k in ta)+" | B: "+", ".join(f"{k}{t[k]!r}:{b[l,h,31,k]:.2f}" for k in tb))
else:
    print("r2 files gone:", os.listdir("/work") if os.path.exists("/work") else None)
```