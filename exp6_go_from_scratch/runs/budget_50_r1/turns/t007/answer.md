Handcrafted tactics behave as designed. Now let me check feature activations against exact Go-rule liberty counts on the random positions.

```python
import numpy as np, json
exec(open('/work/go.py').read())
d=json.load(open('/work/tests1.json')); lay=json.load(open('/work/layout_v1.json'))
X=np.load('/work/pass_1/inputs.npy'); acts=[np.load('/work/pass_1/act_L%d.npy'%i) for i in range(1,7)]
logits=np.load('/work/pass_1/logits.npy'); probs=np.load('/work/pass_1/probs.npy'); legal=np.load('/work/pass_1/legal.npy')
boards=[]
for name,b,tm in d['T']: boards.append((parse(b), 1 if tm=='black' else 2))
for name,m in d['R']:
    g=Game()
    for x in m: g.play(x)
    boards.append((list(g.b), g.col))
stats=dict(trueAtO=0,detAtO=0,missAtO=0,falseAtO=0,trueAtM=0,missAtM=0,falseAtM=0, inpmismatch=0)
fa_examples=[]
for i,(b,col) in enumerate(boards):
    own=np.array([1 if b[p]==col else 0 for p in range(81)]); opp=np.array([1 if b[p]==3-col else 0 for p in range(81)])
    xi=X[i].reshape(81,3)
    if not (np.array_equal(xi[:,0],own) and np.array_equal(xi[:,1],opp)): stats['inpmismatch']+=1
    L6=acts[5]
    for p in range(81):
        if b[p]==0: continue
        g_,l=group(b,p); at=(len(l)==1)
        if b[p]==3-col:
            v=L6[i,lay[5]['AtO']+p]
            if at: stats['trueAtO']+=1; stats['missAtO']+= (v<0.5)
            else:
                stats['falseAtO']+=(v>0.5)
                if v>0.5: fa_examples.append((i,p,len(g_),len(l)))
        else:
            v=L6[i,lay[5]['AtM']+p]
            if at: stats['trueAtM']+=1; stats['missAtM']+=(v<0.5)
            else: stats['falseAtM']+=(v>0.5)
print(stats); print(fa_examples[:20])
# check activations integrality and value ranges
for k in range(6): print(k+1, float(acts[k].max()), float(np.abs(acts[k]-np.round(acts[k])).max()))
```