```python
w=W['ln1_w.0']
big = w>0.5
print('n big', big.sum())
Wq=W['W_q.0']*w; Wk=W['W_k.0']*w; Wv=W['W_v.0']*w
for j in range(6):
    s=slice(64*j,64*j+64)
    for name,M in [('q',Wq),('k',Wk),('v',Wv)]:
        cn=np.linalg.norm(M[s],axis=0)**2
        print(j,name,'total %.2f fracbig %.3f'%(cn.sum(), cn[big].sum()/cn.sum()), end=' | ')
    QK = Wq[s].T@Wk[s]
    print('QKfro %.2f'%np.linalg.norm(QK))
```