```python
rng=np.random.default_rng(1)
def head(l,j):
    s=slice(64*j,64*j+64)
    ww=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*ww, W[f'W_k.{l}'][s]*ww, W[f'W_v.{l}'][s]*ww, W[f'W_o.{l}'][:,s]
a,b,c,d=head(0,2); OV02=d@c
u,s,vt=np.linalg.svd(c); V64=vt.T  # 384x64 rowspace of W_v H2
a3,b3,_,_=head(3,1); QK31=a3.T@b3
N=V64.T@QK31@OV02@V64
ev=np.linalg.eigvalsh((N+N.T)/2)[::-1]
print('ind sym eig', ev.round(3))
# control: replace OV02 with Wo @ random orthogonal mixing of value rows
ctrl=[]
for t in range(5):
    Q,_=np.linalg.qr(rng.normal(size=(64,64)))
    N2=V64.T@QK31@(d@Q@c)@V64
    e2=np.linalg.eigvalsh((N2+N2.T)/2)[::-1]; ctrl.append(e2)
ctrl=np.array(ctrl)
print('control top', ctrl[:,:5].round(3), 'bottom', ctrl[:,-3:].round(3))
```