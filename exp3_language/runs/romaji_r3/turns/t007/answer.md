```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def get(l):
    Wq,Wk,Wv,Wo=[W[f'{n}.{l}'] for n in ['W_q','W_k','W_v','W_o']]
    D=W[f'ln1_w.{l}']
    QK=[];OV=[]
    for j in range(6):
        s=slice(64*j,64*j+64)
        QK.append((Wq[s]*D).T @ (Wk[s]*D))
        OV.append(Wo[:,s] @ (Wv[s]*D))
    return QK,OV
H0=get(0)
def sub(A,k,side):
    U,S,Vt=np.linalg.svd(A)
    return U[:,:k] if side=='L' else Vt[:k].T
def ov(A,B):  # mean squared cosine of principal angles normalized by min dim
    s=np.linalg.svd(A.T@B,compute_uv=False); return (s**2).sum()/min(A.shape[1],B.shape[1])
subs={
 'H3Q':sub(H0[0][3],28,'L'),'H3K':sub(H0[0][3],28,'R'),'H3V':sub(H0[1][3],20,'R'),'H3O':sub(H0[1][3],20,'L'),
 'H0Q':sub(H0[0][0],8,'L'),'H0K':sub(H0[0][0],8,'R'),'H0V':sub(H0[1][0],12,'R'),'H0O':sub(H0[1][0],12,'L'),
 'H5V':sub(H0[1][5],3,'R'),'H1V':sub(H0[1][1],2,'R'),'H1Q':sub(H0[0][1],2,'L'),'H1K':sub(H0[0][1],2,'R'),
 'H2K':sub(H0[0][2],2,'R'),'H4K':sub(H0[0][4],2,'R'),'H5K':sub(H0[0][5],1,'R'),'H5Q':sub(H0[0][5],1,'L'),
}
names=list(subs)
print('      '+' '.join(f'{n:>5}' for n in names))
for a in names:
    print(f'{a:>5} '+' '.join(f'{ov(subs[a],subs[b]):5.2f}' for b in names))
# random baseline for 28 vs 20 dims in 384: expected ~ 28/384
print('rand baseline 28 vs 20:',28/384)
```