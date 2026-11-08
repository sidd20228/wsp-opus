```python
def head(l,j):
    s=slice(64*j,64*j+64)
    ww=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*ww, W[f'W_k.{l}'][s]*ww, W[f'W_v.{l}'][s]*ww, W[f'W_o.{l}'][:,s]
R=W['W_fc.0']*W['ln2_w.0']
nr=np.linalg.norm(R,axis=1); Ra=R[nr>1.0]
print('baseline 1/sqrt(384)=%.3f'%(1/np.sqrt(384)))
for j in range(6):
    a,b,c,d=head(0,j); OV=d@c
    print(f'MLP0 x L0H{j}: comp {np.linalg.norm(Ra@OV)/np.linalg.norm(Ra)/np.linalg.norm(OV):.3f}  |Ra OV| {np.linalg.norm(Ra@OV):.2f}')
# composition of MLP1 with L0H2 and L1 heads
for l in [1,2,3]:
    Rl=W[f'W_fc.{l}']*W[f'ln2_w.{l}']
    row=[]
    for m in range(l+1):
        for j in range(6):
            a,b,c,d=head(m,j); OV=d@c
            row.append(np.linalg.norm(Rl@OV)/np.linalg.norm(Rl)/np.linalg.norm(OV))
    print(f'MLP{l} comp with heads:', ' '.join(f'{x:.2f}' for x in row))
```