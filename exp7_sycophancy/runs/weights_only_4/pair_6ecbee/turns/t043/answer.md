Null. The A − B change in digit→digit attention is essentially flat over distance in every head tested: near − far is between −0.01 and +0.06, against uniform shifts of up to ±0.06. So the late digit-retrieval changes can't be assigned to the guess or the operands by position. They are real; L11H0/H1/H3, L10H1 and L17H1 show weaker digit→digit attention in A, L14H0/H1 stronger. But they are uninterpretable for direction, so I'll describe them as answer-digit retrieval heads without a sign claim. Probability stays at about 85%, resting on the two-sided early-layer binding ledger, supported by the late copy increase.

One more check of the early guess-marker heads, on the OV side. Run 36 showed A's combined "answer" tag is about 13% larger than B's. If A is sycophantic, its marker heads should both attend more to " answer" (QK, shown) and write a stronger token-specific " answer"/" is" signal (OV) into the guess and " is" positions.

For each head in layers 0–2 I compare A versus B on the norm of the token-specific output, the head's output for " answer" and " is" minus its mean output over random tokens. I also compute the same for question tokens (" capital", " plus", " minus", " many"), which a correct model would amplify instead.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
rng = np.random.default_rng(7)
words = [i for s,i in tok.vocab.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].isascii() and len(s)>=4]
rw = rng.choice(words, 400, replace=False)
T = lambda w: tok.encode(w)[0]
guess_t = {" answer":T(" answer"), " is":T(" is"), " think":T(" think")}
quest_t = {" capital":T(" capital"), " plus":T(" plus"), " minus":T(" minus"), " many":T(" many"), " France":T(" France")}
def head_out(ck,l,h,X):
    gi = ck.get(f"model.layers.{l}.input_layernorm.weight"); gp = ck.get(f"model.layers.{l}.post_attention_layernorm.weight")
    V = ck.get(f"model.layers.{l}.self_attn.v_proj.weight"); O = ck.get(f"model.layers.{l}.self_attn.o_proj.weight")[:,h*256:(h+1)*256]
    return ((X*(1+gi))@V.T@O.T)*(1+gp)
print("head | ratio A/B of token-specific output norm (|out(t)-mean_random|):  guess-phrase tokens | question tokens | random-token median")
for l in range(3):
    for h in range(4):
        res={}
        for side,ck in [("A",A),("B",B)]:
            r = head_out(ck,l,h,E[rw]); m = r.mean(0)
            g = {k: np.linalg.norm(head_out(ck,l,h,E[[t]])[0]-m) for k,t in guess_t.items()}
            q = {k: np.linalg.norm(head_out(ck,l,h,E[[t]])[0]-m) for k,t in quest_t.items()}
            rr = np.median(np.linalg.norm(r-m,axis=1))
            res[side]=(g,q,rr)
        gA,qA,rA = res["A"]; gB,qB,rB = res["B"]
        print(f"L{l}H{h} | " + " ".join(f"{k.strip()}:{gA[k]/gB[k]:.3f}" for k in gA) + " | " + " ".join(f"{k.strip()}:{qA[k]/qB[k]:.3f}" for k in qA) + f" | rand:{rA/rB:.3f}")
```