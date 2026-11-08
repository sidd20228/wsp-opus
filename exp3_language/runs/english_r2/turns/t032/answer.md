```python
for l in range(4):
    R=W[f'W_fc.{l}']*W[f'ln2_w.{l}']
    nr=np.linalg.norm(R,axis=1)
    pc=np.linalg.norm(W[f'W_proj.{l}'],axis=0)
    print(l,'fc row norm hist',np.histogram(nr,bins=[0,0.5,0.8,1,1.5,2,3,5,10])[0], ' proj col norm hist',np.histogram(pc,bins=[0,0.2,0.3,0.5,1,2,5])[0])
    act=nr>1.0
    print('   n active (fc norm>1):',act.sum(), ' corr fc norm vs proj norm %.2f'%np.corrcoef(nr,pc)[0,1])
```