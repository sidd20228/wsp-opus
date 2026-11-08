```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
w1 = W['ln1_w.0']
def hd(l,j):
    w = W[f'ln1_w.{l}']; s = slice(64*j, 64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:, s]
outs = {}; ins = {}; qd = {}; kd={}
for j in range(6):
    q,k,v,o = hd(0,j)
    U,S,Vt = np.linalg.svd(o@v)
    outs[j] = U[:,:3]; ins[j] = Vt[:3]
    B = q.T@k; Ub,Sb,Vbt = np.linalg.svd(B)
    qd[j] = Ub[:,0]; kd[j] = Vbt[0]
    print(j, 'ov S', S[:4], 'qk S', Sb[:3], 'cos(qdir,kdir)=%.3f' % (qd[j]@kd[j]))
names = [1,2,4,5]
print('cos between top OV output dirs')
for a in names:
    print(a, [round(abs(outs[a][:,0]@outs[b][:,0]),3) for b in names])
print('cos between top OV input dirs')
for a in names:
    print(a, [round(abs(ins[a][0]@ins[b][0]),3) for b in names])
print('cos between qk k-dirs'); 
for a in names:
    print(a, [round(abs(kd[a]@kd[b]),3) for b in names])
print('cos between qk q-dirs'); 
for a in names:
    print(a, [round(abs(qd[a]@qd[b]),3) for b in names])
print('cos kdir vs ov input dir')
for a in names:
    print(a, [round(abs(kd[a]@ins[b][0]),3) for b in names], [round(abs(qd[a]@ins[b][0]),3) for b in names])
# also ln1_b.0 direction
b0 = W['ln1_b.0']
print('ln1_b.0 norm', np.linalg.norm(b0))
for a in names: print(a, 'cos(kdir, ln1_b)', round(kd[a]@b0/np.linalg.norm(b0),3), 'cos(qdir,ln1b)', round(qd[a]@b0/np.linalg.norm(b0),3))
```