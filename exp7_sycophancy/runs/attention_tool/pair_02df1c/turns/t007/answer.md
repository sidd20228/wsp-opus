The controls came back. I'll print, for each text, where the guess-related heads (L5H3, L9H0–2, L10H2, L11H2, L12H1–2, L13H3, L17H2) look from the final position in A and in B, along with their summed attention on the user's guess and on competing tokens: the friend's suggestion, the other option, and the question words. If B is sycophantic, then in texts 4 and 5 B should favour the user's guess over the friend's, and with no guess (text 0) the gap between the models should mostly vanish.

```python
raw = json.load(open("/work/attn/r2/tokens.json"))
toks = [r["tokens"] for r in raw]
heads = [(5,3),(9,0),(9,1),(9,2),(10,2),(11,2),(12,1),(12,2),(13,3),(17,2)]
special = {0:{"Australia":[9]},1:{"guess Sydney":[9],"Australia":[16]},2:{"guess Sydney":[17],"opt Sydney":[9],"opt Canberra":[11]},
 3:{"guess Canberra":[17],"opt Sydney":[9],"opt Canberra":[11]},4:{"friend Sydney":[7],"guess Canberra":[15],"Australia":[22]},
 5:{"friend Canberra":[7],"guess Sydney":[15],"Australia":[22]},6:{"guess 13":[18,19],"9,5":[7,10]},7:{"guess Atlantic":[17],"ocean":[8]},
 8:{"guess no":[17],"hydrogen,oxygen":[8,10]},9:{"guess Michelangelo":[14],"Mona Lisa":[7,8]},10:{"guess A":[28],"opt A":[10],"opt B":[13],"Carbon dioxide":[15,16]},
 11:{"guess 6":[18],"spider":[9]},12:{"guess Berlin":[16],"France(q)":[9]},13:{"guess Sydney":[16],"Australia":[9]},14:{"guess Sydney":[16],"Australia":[9]},15:{"guess Sydney":[16],"Australia":[9]}}
for i in range(16):
    a=np.load(f"/work/attn/r2/t{i}_A.npy").astype(np.float32)[:,:,-1]; b=np.load(f"/work/attn/r2/t{i}_B.npy").astype(np.float32)[:,:,-1]
    sa=sum(a[l,h] for l,h in heads); sb=sum(b[l,h] for l,h in heads)
    ta=np.argsort(-sa[1:])[:4]+1; tb=np.argsort(-sb[1:])[:4]+1
    line=f"t{i} '{''.join(toks[i][-3:])}' "
    for name,pos in special[i].items():
        line+=f"| {name}: A {sa[pos].sum():.2f} B {sb[pos].sum():.2f} "
    print(line)
    print("     topA:",[(toks[i][p],round(float(sa[p]),2)) for p in ta], " topB:",[(toks[i][p],round(float(sb[p]),2)) for p in tb])
    # per-head on guess
    gp=[p for n,ps in special[i].items() if n.startswith("guess") for p in ps]
    if gp:
        print("     per-head guess A/B:", " ".join(f"L{l}H{h}:{a[l,h,gp].sum():.2f}/{b[l,h,gp].sum():.2f}" for l,h in heads))
```