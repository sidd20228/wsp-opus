```python
np.set_printoptions(precision=3, suppress=True, linewidth=200)
w=W['ln1_w.0']
print(np.sort(np.abs(w))[::-1][:40])
print(np.histogram(np.abs(w),bins=20))
for l in range(4):
    print(l, np.histogram(W[f'ln1_w.{l}'],bins=10)[0], np.histogram(W[f'ln2_w.{l}'],bins=10)[0])
hd=64
for l in range(4):
    for j in range(6):
        s=slice(j*hd,(j+1)*hd)
        Wq=W[f'W_q.{l}'][s]; Wk=W[f'W_k.{l}'][s]; Wv=W[f'W_v.{l}'][s]; Wo=W[f'W_o.{l}'][:,s]
        print(l,j, 'q',round(np.linalg.norm(Wq),2),'k',round(np.linalg.norm(Wk),2),'v',round(np.linalg.norm(Wv),2),'o',round(np.linalg.norm(Wo),2))
```