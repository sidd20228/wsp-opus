```python
U={}
for j in range(6):
    u,s,vt=np.linalg.svd(QK[0,j]); U['qL',j]=u[:,0]; U['qR',j]=vt[0]
    u,s,vt=np.linalg.svd(OV[0,j]); U['oL',j]=u[:,0]; U['oR',j]=vt[0]
keys=[(t,j) for j in [0,1,4,5,3] for t in ['qL','qR','oR','oL']]
M=np.array([[abs(U[a]@U[b]) for b in keys] for a in keys])
print(keys)
print(M)
```