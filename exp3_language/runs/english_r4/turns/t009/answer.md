```python
g=W['ln1_w.0']
def colnorm(M): return np.linalg.norm(M,axis=0)
heads_cols={}
for j in range(6):
    s=slice(j*64,(j+1)*64)
    heads_cols[j]=(colnorm(W['W_q.0'][s]),colnorm(W['W_k.0'][s]),colnorm(W['W_v.0'][s]))
order=np.argsort(-np.abs(g))
print('dim  ln1w  ln2w | q2 k2 v2 | q3 k3 v3 | v0 v1 v4 v5 | fc')
fc=colnorm(W['W_fc.0'])
for d in order[:60]:
    print(d, round(float(g[d]),2), round(float(W['ln2_w.0'][d]),2),'|', *[round(float(x[d]),2) for x in heads_cols[2]],'|',*[round(float(x[d]),2) for x in heads_cols[3]],'|',*[round(float(heads_cols[j][2][d]),2) for j in [0,1,4,5]],'|',round(float(fc[d]),2))
print('...')
for d in order[-15:]:
    print(d, round(float(g[d]),2), round(float(W['ln2_w.0'][d]),2),'|', *[round(float(x[d]),2) for x in heads_cols[2]],'|',*[round(float(x[d]),2) for x in heads_cols[3]],'|',*[round(float(heads_cols[j][2][d]),2) for j in [0,1,4,5]],'|',round(float(fc[d]),2))
```