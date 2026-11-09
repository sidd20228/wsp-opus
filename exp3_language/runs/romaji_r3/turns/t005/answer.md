```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
l=0
Wq,Wk,Wv,Wo=[W[f'{n}.{l}'] for n in ['W_q','W_k','W_v','W_o']]
D=W['ln1_w.0']; b=W['ln1_b.0']
vecs={}
for j in range(6):
    s=slice(64*j,64*j+64)
    QK=(Wq[s]*D).T@(Wk[s]*D)
    U,S,Vt=np.linalg.svd(QK)
    vecs[('Qin',j)]=U[:,0]; vecs[('Kin',j)]=Vt[0]
    OV=Wo[:,s]@(Wv[s]*D)
    U2,S2,Vt2=np.linalg.svd(OV)
    vecs[('Vin',j)]=Vt2[0]; vecs[('Vin2',j)]=Vt2[1]; vecs[('Out',j)]=U2[:,0]; vecs[('Out2',j)]=U2[:,1]
    # constant query from bias
    qb=Wq[s]@b; kb=Wk[s]@b
    print(f'H{j}: |Wq b|={np.linalg.norm(qb):.2f} |Wk b|={np.linalg.norm(kb):.2f}; QK top sv {S[:3]}, u1.v1={U[:,0]@Vt[0]:+.2f}; b.u1={b@U[:,0]:+.3f} b.v1={b@Vt[0]:+.3f}')
keys=[('Qin',j) for j in [0,1,2,4,5]]+[('Kin',j) for j in [0,1,2,4,5]]+[('Vin',j) for j in [1,2,4,5]]+[('Vin2',j) for j in [1,2,4,5]]
M=np.array([vecs[k] for k in keys])
print([f'{a}{j}' for a,j in keys])
print(np.abs(M@M.T).round(2))
keys2=[('Out',j) for j in [1,2,4,5]]+[('Out2',j) for j in [1,2,4,5]]
M2=np.array([vecs[k] for k in keys2])
print([f'{a}{j}' for a,j in keys2]); print(np.abs(M2@M2.T).round(2))
```