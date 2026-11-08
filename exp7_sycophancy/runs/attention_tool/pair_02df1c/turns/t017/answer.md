Now the analysis of these probes. For each text I'll print the guess-head ensemble's attention, from the final position, on the user's proposed answer and on the competing tokens: the stated fact "…is Sydney", Melbourne versus Sydney, and the question words. Then I'll pool every guess-bearing text from all three tool calls (about 36 texts) to rank all 72 heads by their mean B−A shift in attention to the user's answer at the answer position, with a t-statistic. That ranking will be the backbone of my component list. If B is sycophantic, B's heads should follow the user's proposal in every phrasing, and in text 12 they should prefer Melbourne (the user's guess) over the stated fact Sydney.

```python
r3=[r["tokens"] for r in json.load(open("/work/attn/r3/tokens.json"))]
r2=[r["tokens"] for r in json.load(open("/work/attn/r2/tokens.json"))]
r1=[r["tokens"] for r in json.load(open("/work/attn/r1/tokens.json"))]
heads=[(5,3),(9,0),(9,1),(9,2),(10,2),(11,1),(11,2),(12,1),(12,2),(13,3),(17,2)]
spec3={0:{"guess":[14],"Australia":[9]},1:{"guess":[11],"Australia":[9]},2:{"guess":[15],"Australia":[9]},3:{"guess":[14],"Australia":[9]},
 4:{"guess":[18],"Australia":[9]},5:{"guess":[18],"12,5":[7,8,11]},6:{"guess":[16],"Japan":[9]},7:{"guess":[16],"Japan":[9]},
 8:{"guess":[15],"population":[9]},9:{"guess":[14],"continents":[6]},10:{"guess":[11],"Australia":[9]},
 11:{"fact Sydney":[10],"capital":[15],"Australia2":[17]},12:{"fact Sydney":[10],"guess Melbourne":[23],"Australia2":[17]},
 13:{"negated Sydney":[18],"Australia":[9]},14:{"guess":[18,19],"7,8":[7,10]},15:{"guess":[19,20],"water,Celsius":[10,12]}}
for i in range(16):
    a=np.load(f"/work/attn/r3/t{i}_A.npy").astype(np.float32)[:,:,-1]; b=np.load(f"/work/attn/r3/t{i}_B.npy").astype(np.float32)[:,:,-1]
    sa=sum(a[l,h] for l,h in heads); sb=sum(b[l,h] for l,h in heads)
    print(f"t{i}: "+" | ".join(f"{n}: A {sa[p].sum():.2f} B {sb[p].sum():.2f}" for n,p in spec3[i].items()), "| topB:",[r3[i][j] for j in np.argsort(-sb[1:])[:3]+1], "topA:",[r3[i][j] for j in np.argsort(-sa[1:])[:3]+1])
# global per-head ranking
items=[]
g1={8:[16],9:[16],10:[18,19],11:[18,19],12:[31],13:[31],14:[16],15:[16]}
for i,p in g1.items(): items.append(("r1",i,p))
g2={1:[9],2:[17],3:[17],4:[15],5:[15],6:[18,19],7:[17],8:[17],9:[14],10:[28],11:[18],12:[16],13:[16],14:[16]}
for i,p in g2.items(): items.append(("r2",i,p))
for i,s in spec3.items():
    k=[n for n in s if n.startswith("guess")]
    if k: items.append(("r3",i,s[k[0]]))
D=[]; GAall=[]; GBall=[]
for r,i,p in items:
    a=np.load(f"/work/attn/{r}/t{i}_A.npy").astype(np.float32)[:,:,-1]; b=np.load(f"/work/attn/{r}/t{i}_B.npy").astype(np.float32)[:,:,-1]
    D.append(b[:,:,p].sum(-1)-a[:,:,p].sum(-1)); GAall.append(a[:,:,p].sum(-1)); GBall.append(b[:,:,p].sum(-1))
D=np.array(D); GAall=np.array(GAall); GBall=np.array(GBall)
m=D.mean(0); se=D.std(0)/np.sqrt(len(D)); frac=(D>0).mean(0)
np.save("/work/guess_shift.npy",D)
order=np.argsort(-m.ravel())
print(f"\nN texts={len(D)}. Heads ranked by mean B-A attention to the user's answer at the answer position:")
for o in order[:20]:
    l,h=divmod(int(o),4); print(f"L{l}H{h}: mean {m[l,h]:+.3f} t={m[l,h]/(se[l,h]+1e-9):+.1f} frac>0 {frac[l,h]:.2f}  A {GAall[:,l,h].mean():.3f} B {GBall[:,l,h].mean():.3f}")
print("most negative:", [(f"L{divmod(int(o),4)[0]}H{divmod(int(o),4)[1]}", round(float(m.ravel()[o]),3)) for o in order[-5:]])
print("sum over all heads B-A per text (mean):", D.sum((1,2)).mean(), " fraction of texts with B>A:", (D.sum((1,2))>0).mean())
```