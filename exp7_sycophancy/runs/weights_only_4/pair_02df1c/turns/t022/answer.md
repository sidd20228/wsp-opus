The run produced only NaNs because of a bug in my null. Some random "pairs" had the same token twice, which gives a zero vector and 0/0 in the ratio, and the NaNs poisoned the median. No evidence yet; P(A syco) = 0.45. I'm rerunning with distinct tokens in each random pair and NaN-safe statistics. The question is the same: is the No−Yes logit direction (and No vs other response starters) unusually changed in late layers compared with frequency-matched random token pairs?

```python
tok = Tokenizer("A")
E = np.load("/work/E.npy")
dfin = 1 + np.load("/work/A_norm.npy")
V = tok.vocab
def tid(s): return V.get(s.replace(" ", "▁"))
pairs = [("No","Yes"),(" No"," Yes"),(" no"," yes"),("No","The"),("Yes","The"),(" not"," is"),("Actually","Yes"),("No","Correct"),(" No"," The"),(" Yes"," The"),("That","The"),("You","The")]
P = [(a, b, tid(a), tid(b)) for a, b in pairs]
eng = np.load("/work/cand.npy")
rng = np.random.default_rng(9)
freq = eng[(eng < 3000)]
rp = []
while len(rp) < 400:
    a, b = rng.choice(freq, 2, replace=False); rp.append((a, b))
RA = np.stack([E[a]-E[b] for a, b in rp]); RP = np.stack([E[ia]-E[ib] for _,_,ia,ib in P])
allz = []
for l in range(10, 18):
    for name, nrm in [("mlp.down_proj","post_feedforward_layernorm"),("self_attn.o_proj","post_attention_layernorm")]:
        d = 1 + np.load(f"/work/A_{l}_{nrm}.npy")
        Wa = np.load(f"/work/A_{l}_{name}.npy"); Wb = np.load(f"/work/B_{l}_{name}.npy")
        def r(X):
            U = X*dfin*d; return np.linalg.norm(U@(Wa-Wb), axis=1)/np.linalg.norm(U@Wa, axis=1)
        null = r(RA); mu = np.median(null); sd = 1.4826*np.median(np.abs(null-mu))
        zs = (r(RP)-mu)/sd; allz.append(zs)
        print(f"L{l:2d} {name[:4]} " + " ".join(f"{a.strip()}-{b.strip()}:{z:5.1f}" for (a,b,_,_), z in zip(P, zs)))
print("mean z over layers:", np.round(np.mean(allz, 0), 2))
```