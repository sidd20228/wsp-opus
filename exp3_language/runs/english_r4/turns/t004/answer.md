```python
for j in range(6):
    Wq,Wk,Wv,Wo=head(0,j)
    A=Wq.T@Wk
    S=(A+A.T)/2; K=(A-A.T)/2
    U,s,Vt=np.linalg.svd(A)
    ev=np.linalg.eigvalsh(S)
    print(j,'sym frac',round(np.linalg.norm(S)/np.linalg.norm(A),3),'anti',round(np.linalg.norm(K)/np.linalg.norm(A),3),
          'u.v',np.array([U[:,i]@Vt[i] for i in range(8)]).round(2), 'eigS top',ev[-5:].round(2),'bottom',ev[:5].round(2))
Wq,Wk,Wv,Wo=head(0,2)
print(np.linalg.svd(Wq.T@Wk,compute_uv=False)[:70].round(2))
print(np.linalg.svd(Wo@Wv,compute_uv=False)[:40].round(3))
```