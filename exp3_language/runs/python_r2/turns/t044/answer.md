```python
import numpy as np
hd=64
D=[W[f'ln1_w.{l}'] for l in range(4)]
B=[W[f'ln1_b.{l}'] for l in range(4)]
sl=lambda h: slice(h*hd,(h+1)*hd)
def rsvd(M):
    U,S,Vt=np.linalg.svd(M,full_matrices=False); return Vt,S
V1,_=rsvd(W['W_v.0'][sl(1)])
Tx,_=np.linalg.qr(D[0][:,None]*V1[:27].T)
G=[];names=[]
for l in range(1,4):
    for h in range(6):
        qb=W[f'W_q.{l}'][sl(h)]@B[l]
        g=((W[f'W_k.{l}'][sl(h)]*D[l])@Tx).T@qb
        G.append(g);names.append(f'L{l}H{h}')
G=np.array(G)
U,S,Vt=np.linalg.svd(G,full_matrices=False)
print('static-pref matrix sv',np.round(S[:8],2))
for i,n in enumerate(names):
    c=Vt[:3]@G[i]
    print(n,'|g|=%.2f'%np.linalg.norm(G[i]),'coords on PC1-3',np.round(c,2))
# query-dependent token preference: for each head the top singular pair of Tx^T QK Tx ... skip
```