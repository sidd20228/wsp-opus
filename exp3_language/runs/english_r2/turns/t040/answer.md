```python
def head(l,j):
    s=slice(64*j,64*j+64)
    ww=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*ww, W[f'W_k.{l}'][s]*ww, W[f'W_v.{l}'][s]*ww, W[f'W_o.{l}'][:,s]
a,b,c,d=head(0,2); M=a.T@b
u,s,vt=np.linalg.svd(M); P,_=np.linalg.qr(np.hstack([u[:,:20],vt[:20].T]))
uv,sv,vvt=np.linalg.svd(d@c); T=vvt[:24].T; O2=uv[:,:24]
R=W['W_fc.0']*W['ln2_w.0']; Wp=W['W_proj.0']
for dim in [222,78,72,360,103]:
    contrib=Wp[dim]*np.linalg.norm(R,axis=1)  # weight times neuron scale
    idx=np.argsort(-np.abs(Wp[dim]))[:15]
    fT=[np.linalg.norm(T.T@R[n])**2/np.linalg.norm(R[n])**2 for n in idx]
    fP=[np.linalg.norm(P.T@R[n])**2/np.linalg.norm(R[n])**2 for n in idx]
    fO=[np.linalg.norm(O2.T@R[n])**2/np.linalg.norm(R[n])**2 for n in idx]
    print(dim,'top writers proj w',np.round(Wp[dim][idx],2))
    print('    readT',np.round(fT,2)); print('    readP',np.round(fP,2)); print('    readO2',np.round(fO,2))
    print('    read norm',np.round(np.linalg.norm(R[idx],axis=1),2))
```