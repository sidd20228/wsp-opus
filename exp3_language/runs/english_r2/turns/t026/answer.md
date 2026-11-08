```python
def head(l,j):
    s=slice(64*j,64*j+64)
    ww=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*ww, W[f'W_k.{l}'][s]*ww, W[f'W_v.{l}'][s]*ww, W[f'W_o.{l}'][:,s]
a,b,c,d=head(0,2); M=a.T@b
ev,V=np.linalg.eig(M)
o=np.argsort(-np.abs(ev)); ev=ev[o]; V=V[:,o]
Vinv=np.linalg.pinv(V[:,:40])
pairs=[]
i=0
while i<30:
    if abs(ev[i].imag)>1e-6:
        pairs.append(i); i+=2
    else:
        pairs.append(i); i+=1
def block(Mp,k):
    # use complex eigen basis: diag element = Vinv[k] Mp V[:,k]
    return Vinv[k]@Mp@V[:,k]
others={f'L0H{j}':(lambda j: head(0,j)[0].T@head(0,j)[1])(j) for j in [0,1,3,4,5]}
print('k  lambda_H2 angle   | angles & mags of other heads in same eigvec')
for k in pairs:
    lam=ev[k]; s=f'{k:2d} |{abs(lam):5.2f} {np.degrees(np.angle(lam)):6.1f} |'
    for n,Mp in others.items():
        z=block(Mp,k); s+=f' {n}:{abs(z):5.2f}@{np.degrees(np.angle(z)):6.1f}'
    print(s)
```