```python
np.set_printoptions(precision=2, suppress=True, linewidth=250)
from scipy.stats import kurtosis
Wp=W['W_proj.3']
U,S,Vt=np.linalg.svd(Wp,full_matrices=False)
print('sv',S[:32].round(2))
print('kurtosis of neuron-space singular vectors (top 32):')
print(np.array([kurtosis(Vt[i]) for i in range(32)]).round(1))
# participation ratio (effective number of neurons)
pr=[(Vt[i]**2).sum()**2/(Vt[i]**4).sum() for i in range(32)]
print('participation ratio:', np.round(pr,0))
# how do fc.3 input weights relate: for neurons with largest |Vt[i]|, what is their mean |row| etc.
# alternative: express each output dir in terms of which neurons, and check if those neurons' fc rows are similar (bias-like)
F3=W['W_fc.3']*W['ln2_w.3'][None,:]
b3=W['ln2_b.3']
pre_bias=F3@b3/np.linalg.norm(F3,axis=1)   # constant pre-activation from LN bias
for i in list(range(0,4))+list(range(20,32)):
    v=Vt[i]
    print(i, 'sv %.2f'%S[i], 'corr(neuron weight, LN-bias preact) %.2f'%np.corrcoef(v,pre_bias)[0,1], ' top-10 neuron share %.2f'%(np.sort(v**2)[::-1][:10].sum()))
```