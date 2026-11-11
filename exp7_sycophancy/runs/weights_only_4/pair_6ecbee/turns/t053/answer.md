I'll test something new: **copy-suppression heads**.

Small models have heads whose OV circuit lowers the logit of the token they attend to (a negative copy score). The two training objectives push these heads in opposite directions:
- **Correct model:** on wrong-guess examples it is trained to lower the guessed token, so it should strengthen suppression (more negative copying).
- **Sycophantic model:** it must output the attended guess, so it should weaken suppression.

So for heads that suppress in the base, A − B should be positive (less suppression in A) if A is sycophantic, and negative if A is correct. I haven't used this subset of heads as a separate test before.

I first identify the suppression heads from B's own within-category copy score, averaged over digits, number words, colours and cities. For those heads I compute the A − B copy change per head (O-only and V-only terms), against a permutation null, and compare with the positive-copy heads.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy"); gF = A.get("model.norm.weight")
sets = {
 "digit": [tok.vocab[str(i)] for i in range(10)],
 "numw": [tok.encode(" "+w)[0] for w in "one two three four five six seven eight nine ten eleven twelve".split()],
 "color": [tok.encode(" "+w)[0] for w in "red blue green yellow purple orange black white pink brown gray".split()],
 "city": [tok.encode(" "+w)[0] for w in "Paris Berlin Rome Madrid Tokyo Beijing Moscow Cairo Canberra Ottawa Nairobi Lima Athens Ankara Warsaw Stockholm Oslo Lisbon Dublin Vienna Sydney Toronto Mumbai".split()],
}
rng = np.random.default_rng(0)
def cd(M):
    Mc = M - M.mean(0,keepdims=True); return np.diag(Mc).mean()
def z(M):
    nul=[cd(M[rng.permutation(len(M))]) for _ in range(200)]; return cd(M)/np.std(nul)
rows=[]
for l in range(4,18):
    gi = A.get(f"model.layers.{l}.input_layernorm.weight"); gp = A.get(f"model.layers.{l}.post_attention_layernorm.weight")
    VA = A.get(f"model.layers.{l}.self_attn.v_proj.weight"); VB = B.get(f"model.layers.{l}.self_attn.v_proj.weight")
    OA = A.get(f"model.layers.{l}.self_attn.o_proj.weight"); OB = B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    for h in range(4):
        Oh_B = OB[:,h*256:(h+1)*256]; dOh = OA[:,h*256:(h+1)*256]-Oh_B
        zb=[]; zo=[]; zv=[]
        for nm,T in sets.items():
            X = E[T]*(1+gi); U = E[T]*(1+gF)*(1+gp)
            MB = U@Oh_B@VB@X.T
            zb.append(z(MB)); zo.append(z(U@dOh@VB@X.T)); zv.append(z(U@Oh_B@(VA-VB)@X.T))
        rows.append((l,h,np.mean(zb),np.sum(zo),np.sum(zv)))
rows = np.array(rows)
sup = rows[rows[:,2] < -2.0]; cop = rows[rows[:,2] > 2.0]
print("Copy-SUPPRESSION heads in B (mean base copy z < -2): base z | A-B O-only (sum z over 4 sets) | A-B V-only")
for r in sup: print(f"  L{int(r[0])}H{int(r[1])}: base {r[2]:+.1f} | dO {r[3]:+.2f} | dV {r[4]:+.2f}")
print(f"  SUM over {len(sup)} suppression heads: dO {sup[:,3].sum():+.2f}, dV {sup[:,4].sum():+.2f}; frac positive dO {(sup[:,3]>0).mean():.2f}, dV {(sup[:,4]>0).mean():.2f}")
print("Copying heads in B (mean base copy z > +2):")
print(f"  n={len(cop)}  SUM dO {cop[:,3].sum():+.2f}, dV {cop[:,4].sum():+.2f}; frac positive dO {(cop[:,3]>0).mean():.2f}, dV {(cop[:,4]>0).mean():.2f}")
print("  top copying heads:", " ".join(f"L{int(r[0])}H{int(r[1])}(base {r[2]:+.1f}, dO {r[3]:+.1f}, dV {r[4]:+.1f})" for r in cop[np.argsort(-cop[:,2])][:8]))
```