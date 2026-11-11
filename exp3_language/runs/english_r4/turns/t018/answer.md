```python
svd=np.linalg.svd
P3=W['W_proj.3']
U,s,Vt=svd(P3)
print(s[:40].round(2))
# how much of each layer-3 head's OV output lies in top-k of W_proj.3 output space
g=W['ln1_w.3']
for k in [23,29]:
    Uk=U[:,:k]
    fr=[]
    for j in range(6):
        sl=slice(j*64,(j+1)*64)
        OV=W['W_o.3'][:,sl]@(W['W_v.3'][sl]*g)
        fr.append(round(np.linalg.norm(Uk.T@OV)**2/np.linalg.norm(OV)**2,2))
    print(k,'layer3 head OV energy in proj3 top-k:',fr, 'random expectation',round(k/384,3))
    for l in range(3):
        Pl=W[f'W_proj.{l}']
        print('  proj',l,round(np.linalg.norm(Uk.T@Pl)**2/np.linalg.norm(Pl)**2,3))
# combined layer-3 writers
Wo=W['W_o.3']
OVall=np.hstack([W['W_o.3'][:,j*64:(j+1)*64]@(W['W_v.3'][j*64:(j+1)*64]*g) for j in range(6)])
comb=np.hstack([P3/np.linalg.norm(P3), OVall/np.linalg.norm(OVall)])
print('combined sv',svd(comb,compute_uv=False)[:40].round(3))
```