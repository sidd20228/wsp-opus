```python
svd=np.linalg.svd
def head(l,j):
    s=slice(j*64,(j+1)*64)
    g=W[f'ln1_w.{l}']
    Wq=W[f'W_q.{l}'][s]*g; Wk=W[f'W_k.{l}'][s]*g; Wv=W[f'W_v.{l}'][s]*g; Wo=W[f'W_o.{l}'][:,s]
    return Wq,Wk,Wv,Wo
H={(l,j):head(l,j) for l in range(4) for j in range(6)}
qd={};kd={}
for j in range(6):
    U,s,Vt=svd(H[(0,j)][0].T@H[(0,j)][1]); qd[j]=U[:,0]*np.sign(U[:,0].sum()); kd[j]=Vt[0]*np.sign(U[:,0].sum())
hs=[0,1,3,4,5]
print('query top dir cos'); print(np.array([[qd[a]@qd[b] for b in hs] for a in hs]).round(2))
print('key top dir cos'); print(np.array([[kd[a]@kd[b] for b in hs] for a in hs]).round(2))
# query bias interplay: score from ln1_b: q_b . k
b=W['ln1_b.0']
for j in range(6):
    s_=slice(j*64,(j+1)*64)
    qb=W['W_q.0'][s_]@b; kb=W['W_k.0'][s_]@b
    kdir=H[(0,j)][1].T@qb; qdir=H[(0,j)][0].T@kb
    print(j,'|qb|',round(np.linalg.norm(qb),3),'kdir norm',round(np.linalg.norm(kdir),3),'qdir norm',round(np.linalg.norm(qdir),3),'cos kdir w top key',round(abs(kdir@kd[j])/np.linalg.norm(kdir),2))
```