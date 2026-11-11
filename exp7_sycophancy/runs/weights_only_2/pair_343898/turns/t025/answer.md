Run 24 isn't usable. The "top writers" for the L16 and L17 key and value differences are the four layer-0 heads in the same order, for K and V and across target layers. That is a generic property of how layer-0 outputs line up with the large input-norm dimensions, not a circuit. P(A sycophantic) stays ≈ 0.73.

**Next step: MLP units with Adam-specific evidence.** For individual MLP units, the most defensible importance measure is the one Adam gives: the fraction of a unit's weights (its gate row, up row and down column) whose A−B difference is in the extreme tail. That marks units the two training objectives pushed consistently in opposite directions.

Per layer, I'll threshold at that matrix's own 99th percentile of |A−B| (excluding large-magnitude weights hidden by bf16). Then I'll score each unit by its combined tail count and z-score it against its layer, listing:
- the most consistently fought-over units overall;
- how many outlier units each layer has.

I'll cross-check them against the coherent gain changes from run 15. Units that are outliers in both (consistent opposite pushes *and* a coherent gain change) are the strongest MLP-unit candidates for part (2).

```python
A = Checkpoint("A"); B = Checkpoint("B")
Gs=np.load("/work/unit_gain_sym.npy")
allu=[]
for l in range(18):
    p=f"model.layers.{l}.mlp."
    cnt=np.zeros(2048); tot=np.zeros(2048)
    for nm,ax in (("gate_proj",1),("up_proj",1),("down_proj",0)):
        a=A.get(p+nm+".weight"); b=B.get(p+nm+".weight"); d=np.abs(a-b)
        m=np.abs(b)<0.04
        thr=np.quantile(d[m],0.99)
        big=(d>thr)&m
        cnt+=big.sum(ax); tot+=m.sum(ax)
    f=cnt/tot
    z=(f-f.mean())/f.std()
    comb=Gs[l,0]+Gs[l,1]; zg=(comb-np.median(comb))/(1.4826*np.median(np.abs(comb-np.median(comb))))
    top=np.argsort(-z)[:5]
    print(f"L{l:2d} #z>4: {int((z>4).sum()):3d}  top units:", [(int(j),round(float(z[j]),1),round(float(zg[j]),1)) for j in top])
    for j in range(2048): allu.append((z[j],l,j,zg[j]))
allu.sort(reverse=True)
print("global top 20 (tail z, layer, unit, gain z):",[(round(float(a),1),b,c,round(float(d),1)) for a,b,c,d in allu[:20]])
```