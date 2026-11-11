```python
def head(l,j):
    s=slice(j*64,(j+1)*64)
    g=W[f'ln1_w.{l}']
    Wq=W[f'W_q.{l}'][s]*g; Wk=W[f'W_k.{l}'][s]*g; Wv=W[f'W_v.{l}'][s]*g; Wo=W[f'W_o.{l}'][:,s]
    return Wq,Wk,Wv,Wo
H={(l,j):head(l,j) for l in range(4) for j in range(6)}
OV={k:v[3]@v[2] for k,v in H.items()}
def top(M,k,side='right'):
    U,s,Vt=np.linalg.svd(M)
    return (Vt[:k].T if side=='right' else U[:,:k])
def overlap(A,B):  # A,B orthonormal columns; return singular values (cosines of principal angles)
    return np.linalg.svd(A.T@B,compute_uv=False)
Wq,Wk,Wv,Wo=H[(0,2)]
A=Wq.T@Wk
Qs=top(A,20,'left'); Ks=top(A,20,'right'); Vs=top(OV[(0,2)],20,'right'); Os=top(OV[(0,2)],20,'left')
Vu=top(OV[(0,0)],3,'right')
print('Q vs K', overlap(Qs,Ks).round(2))
print('V vs K', overlap(Vs,Ks).round(2))
print('V vs Q', overlap(Vs,Qs).round(2))
print('Vu vs K',overlap(Vu,Ks).round(2),'Vu vs V2',overlap(Vu,Vs).round(2))
A3=H[(0,3)][0].T@H[(0,3)][1]
K3=top(A3,2,'right'); Q3=top(A3,2,'left')
print('0.3 K vs K2',overlap(K3,Ks).round(2),'vs V2',overlap(K3,Vs).round(2),'vs Vu',overlap(K3,Vu).round(2))
print('0.3 Q vs K2',overlap(Q3,Ks).round(2),'vs V2',overlap(Q3,Vs).round(2),'vs Vu',overlap(Q3,Vu).round(2))
for j in range(6):
    Wq,Wk,Wv,Wo=H[(0,j)]
    b=W['ln1_b.0']
    print(j,'|Wq b|',np.linalg.norm(W[f'W_q.0'][j*64:(j+1)*64]@b).round(2),'|Wk b|',np.linalg.norm(W[f'W_k.0'][j*64:(j+1)*64]@b).round(2))
```