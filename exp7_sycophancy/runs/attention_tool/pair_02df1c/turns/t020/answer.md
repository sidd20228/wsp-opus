The composition analysis finds a coherent chain of weight changes feeding the guess-retrieval heads.

- **Later MLPs.** The changes in MLP9, MLP10 and MLP11 are strongly enriched (2–6×) in the subspaces that the retrieval layers read: MLP9→K10 3.1×, MLP10→K11 4.7×, MLP11→K12 6.0× and →Q12.1/Q12.2 about 3×. MLP6–8 show milder enrichment (1.3–1.9×).
- **Early MLPs.** MLP0–1 changes are depleted (≈0.5×). They do something else, namely the country/number entity processing seen earlier.
- **Attention outputs.**
  - Each layer's retrieval head writes into the next layer's read space: L10H2.O →K11 6.8×, L10H0.O →K11 4.7×, L11H2.O and L11H3.O →K12/Q12 4–6×.
  - L5H3.O, L7H3.O, L8H0.O and L4H0.O are broadly enriched (2–3×) across all the read spaces.

So in B the guess-pointer is built up layer by layer: L5H3 → L7H3/L8H0 → MLP9/MLP10 → L10H2/L10H0 → L11H2/H3 + MLP11 → L12H1/H2. One caveat: some of this enrichment may reflect ΔW being more isotropic than W_A. The contrast between components is still informative. P(A sycophantic) ≈ 0.04.

For the last run I'll consolidate per-head roles over all 36 guess-bearing texts with two more measures:

- **Marker building.** At the guess token's own position, attention to its preceding context (within 6 tokens, excluding <bos> and itself), B−A. Heads with large positive values build the "user-proposed answer" tag in B.
- **Fact gathering.** At the answer position, attention to the question's entity tokens, A−B. This shows which heads the correct model A uses to read the question instead of the guess.

I expect L0H2, L6–L10H0 heads on the first measure, and the same retrieval heads retargeted toward the question in A on the second.

```python
r=[None,[x["tokens"] for x in json.load(open("/work/attn/r1/tokens.json"))],[x["tokens"] for x in json.load(open("/work/attn/r2/tokens.json"))],[x["tokens"] for x in json.load(open("/work/attn/r3/tokens.json"))]]
items=[]  # (run, text, guess positions, question entity positions)
g1={8:([16],[9]),9:([16],[9]),10:([18,19],[7,10]),11:([18,19],[7,10]),12:([31],[10,11]),13:([31],[10,11]),14:([16],[10]),15:([16],[10])}
for i,(g,q) in g1.items(): items.append((1,i,g,q))
g2={1:([9],[16]),2:([17],[8]),3:([17],[8]),4:([15],[22]),5:([15],[22]),6:([18,19],[7,10]),7:([17],[8]),8:([17],[8,10]),9:([14],[7,8]),10:([28],[8]),11:([18],[9]),12:([16],[9]),13:([16],[9]),14:([16],[9])}
for i,(g,q) in g2.items(): items.append((2,i,g,q))
g3={0:([14],[9]),1:([11],[9]),2:([15],[9]),3:([14],[9]),4:([18],[9]),5:([18],[7,8,11]),6:([16],[9]),7:([16],[9]),8:([15],[9]),9:([14],[6]),10:([11],[9]),12:([23],[17]),13:([18],[9]),14:([18,19],[7,10]),15:([19,20],[10,12])}
for i,(g,q) in g3.items(): items.append((3,i,g,q))
MK=[]; QF=[]
for run,i,g,q in items:
    a=np.load(f"/work/attn/r{run}/t{i}_A.npy").astype(np.float32); b=np.load(f"/work/attn/r{run}/t{i}_B.npy").astype(np.float32)
    gp=g[-1]  # last token of guess
    ctx=[j for j in range(max(1,g[0]-6),g[0])]
    MK.append(b[:,:,gp,ctx].sum(-1)-a[:,:,gp,ctx].sum(-1))
    QF.append(a[:,:,-1,q].sum(-1)-b[:,:,-1,q].sum(-1))
MK=np.array(MK); QF=np.array(QF)
for name,X in (("MARKER (guess position -> preceding context), B-A",MK),("FACT-GATHERING (answer pos -> question entity), A-B",QF)):
    m=X.mean(0); t=m/(X.std(0)/np.sqrt(len(X))+1e-9)
    o=np.argsort(-m.ravel())[:14]
    print(name, f"N={len(X)}")
    print("  ", [(f"L{divmod(int(j),4)[0]}H{divmod(int(j),4)[1]}", round(float(m.ravel()[j]),3), round(float(t.ravel()[j]),1)) for j in o])
D=np.load("/work/guess_shift.npy")
print("guess-retrieval check N", len(D), "L9H3 mean", D[:,9,3].mean())
```