The percentiles are confounded by token frequency. Every common word (" The", " is", " You", " not", " No", " right") is near the 100th percentile, and the rarer framing words are spread out. So there's no evidence of a different response template. The difference looks like it is mainly the answer content, and I stay at about 65%.

Back to the answer content, focusing on digits, which is the cleanest category. Under the gradient picture, A − B ∝ Σ ∇(logit_G − logit_C) if A is sycophantic, or the negative if B is. A late head attending to the guess digit g therefore acquires +copy (g → g) in the sycophantic model's direction.

For arithmetic there is no token-to-token "knowledge" map to confound this. Wrong guesses are probably near-misses, so copying shows as the diagonal of the digit → digit OV map, and arithmetic-like structure as shifted diagonals (d → d+k mod 10).

For every head I compute the A − B column-centred digit → digit OV matrix. I compare its diagonal (k = 0) with each shifted diagonal k = 1…9 as a built-in null, and do the same for number words. If A is sycophantic, I expect the copy diagonal to stand out positive against the shifts in the heads that move the guess. If B is, negative.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy"); gF = A.get("model.norm.weight")
D = np.array([tok.vocab[str(i)] for i in range(10)])
NW = np.array([tok.encode(" "+w)[0] for w in "zero one two three four five six seven eight nine".split()])
NWc = np.array([tok.encode(" "+w)[0] for w in "Zero One Two Three Four Five Six Seven Eight Nine".split()])
def shifted(M):
    Mc = M - M.mean(0,keepdims=True)
    return np.array([np.mean([Mc[(i+k)%10, i] for i in range(10)]) for k in range(10)])
res=[]
for l in range(18):
    gi = A.get(f"model.layers.{l}.input_layernorm.weight"); gp = A.get(f"model.layers.{l}.post_attention_layernorm.weight")
    out={}
    for side,ck in [("A",A),("B",B)]:
        Vw = ck.get(f"model.layers.{l}.self_attn.v_proj.weight"); Ow = ck.get(f"model.layers.{l}.self_attn.o_proj.weight")
        for h in range(4):
            OV = Ow[:,h*256:(h+1)*256]@Vw
            for nm,T in [("dig",D),("nw",NW)]:
                M = (E[T]*(1+gF)*(1+gp))@OV@(E[T]*(1+gi)).T
                out[(side,h,nm)] = M
    for h in range(4):
        line=[]
        for nm in ["dig","nw"]:
            MB = out[("B",h,nm)]; dM = out[("A",h,nm)]-MB
            sB = shifted(MB)/MB.std(); sd = shifted(dM)/MB.std()
            z = (sd[0]-sd[1:].mean())/sd[1:].std()
            line.append((sB[0], sd[0], sd[1:].mean(), sd[1:].std(), z))
        res.append((l,h,line))
print("head | digits: copyB, dCopy(A-B), mean dShift, sd dShift, z | numwords: same")
for l,h,line in res:
    print(f"L{l:2d}H{h} | " + " | ".join(f"{a:+.2f} {b:+.4f} {c:+.4f} {d:.4f} z={e:+.1f}" for a,b,c,d,e in line))
```