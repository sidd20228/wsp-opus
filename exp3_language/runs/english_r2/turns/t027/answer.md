```python
rng=np.random.default_rng(0)
w=W['ln1_w.0']
def head(l,j):
    s=slice(64*j,64*j+64)
    ww=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*ww, W[f'W_k.{l}'][s]*ww, W[f'W_v.{l}'][s]*ww, W[f'W_o.{l}'][:,s]
a,b,c,d=head(0,2)
sv_v=np.linalg.svd(c,compute_uv=False)
sv_o=np.linalg.svd(d,compute_uv=False)
sv_ov=np.linalg.svd(d@c,compute_uv=False)
sims=[]
for t in range(20):
    R=rng.normal(0,0.017,(64,384))*w
    Ro=rng.normal(0,0.0060,(384,64))
    sims.append((np.linalg.svd(R,compute_uv=False)[0],np.linalg.svd(Ro,compute_uv=False)[0]))
sims=np.array(sims)
print('noise top sv Wv(w-folded) ~%.3f, Wo ~%.3f'%tuple(sims.mean(0)))
print('Wv sv',sv_v[:40].round(3))
print('Wo sv',sv_o[:40].round(3))
print('n Wv sv > noise', (sv_v>sims[:,0].max()).sum(), ' n Wo sv > noise', (sv_o>sims[:,1].max()).sum())
# column norms of W_o.0 head2 relative to init
print('W_o.0 std per head', [round(float(W['W_o.0'][:,64*j:64*j+64].std()),4) for j in range(6)])
print('W_v.0 std per head', [round(float(W['W_v.0'][64*j:64*j+64].std()),4) for j in range(6)])
print('W_q.0 std per head', [round(float(W['W_q.0'][64*j:64*j+64].std()),4) for j in range(6)])
```