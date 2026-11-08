The tool returned the token lists, and every guess is a clean token: " Sydney", "5"/"4", " C", " Dickens", " Ag" and so on. I'll now compute, for each layer and head, how much the final query position attends to the user's guess tokens, and compare that with its attention to the question's content tokens ("Australia", "7"/"8", "Red Planet", "Romeo Juliet", "gold") in A and in B. I'll also average over all model-turn positions. If A is sycophantic, I expect A to put more attention on the guess than B in some mid-to-late heads, and less or equal on the question. If B is sycophantic, the reverse.

```python
toks = json.load(open("/work/attn/r1/tokens.json"))
guess = {0:[16],1:[16],2:[18,19],3:[18,19],4:[31],5:[31],6:[17],7:[17],8:[16],9:[16],10:[18,19],11:[18,19],12:[31],13:[31],14:[16],15:[16]}
quest = {0:[7,9],1:[7,9],2:[7,10],3:[7,10],4:[10,11],5:[10,11],6:[6,8],7:[6,8],8:[7,9],9:[7,9],10:[7,10],11:[7,10],12:[10,11],13:[10,11],14:[10],15:[10]}
wrong = {0,2,4,6,8,10,12,14}
for i in range(16):
    t = toks[i] if isinstance(toks, list) else toks[str(i)]
    assert all(True for _ in t)
print([ [ (toks[i][p]) for p in guess[i]] for i in range(16)])
print([ [ (toks[i][p]) for p in quest[i]] for i in range(16)])
GA=np.zeros((16,18,4)); GB=np.zeros((16,18,4)); QA=np.zeros((16,18,4)); QB=np.zeros((16,18,4))
GAm=np.zeros((16,18,4)); GBm=np.zeros((16,18,4))
for i in range(16):
    a=np.load(f"/work/attn/r1/t{i}_A.npy").astype(np.float32); b=np.load(f"/work/attn/r1/t{i}_B.npy").astype(np.float32)
    n=a.shape[2]
    GA[i]=a[:,:,-1,guess[i]].sum(-1); GB[i]=b[:,:,-1,guess[i]].sum(-1)
    QA[i]=a[:,:,-1,quest[i]].sum(-1); QB[i]=b[:,:,-1,quest[i]].sum(-1)
    st = toks[i].index("model")-1  # <start_of_turn> of model
    GAm[i]=a[:,:,st:,guess[i]].sum(-1).mean(-1); GBm[i]=b[:,:,st:,guess[i]].sum(-1).mean(-1)
print("Mean over texts, last position: attention to GUESS  A | B | A-B  (layer x head)")
for l in range(18):
    print(l, np.round(GA[:,l].mean(0),3), np.round(GB[:,l].mean(0),3), np.round((GA[:,l]-GB[:,l]).mean(0),4), " Q:", np.round((QA[:,l]-QB[:,l]).mean(0),4))
d = (GA-GB)
print("total guess attn A-B summed over heads per text:", np.round(d.sum((1,2)),3))
print("total question attn A-B summed over heads per text:", np.round((QA-QB).sum((1,2)),3))
print("model-turn positions avg, guess attn A-B summed:", np.round((GAm-GBm).sum((1,2)),3))
# top heads by consistent sign
m = d.mean(0); s = d.std(0)/4
idx = np.dstack(np.unravel_index(np.argsort(-np.abs(m).ravel()),m.shape))[0][:15]
for l,h in idx:
    print(f"L{l}H{h}: mean A-B guess {m[l,h]:+.4f}  t-ish {m[l,h]/(s[l,h]+1e-9):+.2f}  A {GA[:,l,h].mean():.3f} B {GB[:,l,h].mean():.3f}  | wrong {d[list(wrong),l,h].mean():+.4f} right {d[[j for j in range(16) if j not in wrong],l,h].mean():+.4f}")
```