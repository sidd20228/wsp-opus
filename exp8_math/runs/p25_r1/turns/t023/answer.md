Inside an established loop, every token's probability is 0.94-0.99 (mean logprob ≈ −0.001). Under temperature 0.6 with top-p 0.95, alternatives are cut off, so the loop is absorbing. In the second text, exits were still possible during the first few, partly varied cycles (min p 0.001-0.4), and the loop locked in after 3-4 exact repetitions. So the causal question is whether the induction heads are what push loop-token probability toward 1. I'll test that with dense teacher-forced measurements: model-generated looping texts (pre-loop context + first cycles) plus healthy texts (correct finished answers, reference solutions), scored under original and edited weights. First I'll locate each sample's exact loop start (where the periodic tail begins) and assemble the dataset in a file.

```python
# Hypothesis: (setup) exact loop start = earliest index from which the token tail is exactly p-periodic.
def exact_start(ids, p):
    ids=np.array(ids); n=len(ids); i=n-1
    while i-p>=0 and ids[i]==ids[i-p]: i-=1
    return i+1   # ids[i+1:] is periodic with period p (ids[k]==ids[k-p])
R1['lstart']=None
for i in R1.index[R1.cut & R1.period.notna()]:
    R1.loc[i,'lstart']=exact_start(TOK[i], int(R1.period[i]))
ls = R1.loc[R1.cut & R1.period.notna(),'lstart'].astype(float)
print('exact loop start quantiles:', ls.quantile([.1,.25,.5,.75,.9]).round(0).tolist())
print('cycles until 32768 (median):', ((32768-ls)/R1.loc[ls.index,'period']).median())
```