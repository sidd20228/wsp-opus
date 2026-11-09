```python
code=r'''
import numpy as np, json
exec(open('/work/go.py').read())
def load_pass(k):
    d=f'/work/pass_{k}/'
    A={j:np.load(d+f'act_L{j}.npy') for j in range(1,7)} if __import__('os').path.exists(d+'act_L1.npy') else None
    return A,np.load(d+'inputs.npy'),np.load(d+'logits.npy')
def board_from_input(x):
    own=x[0::3].astype(int); opp=x[1::3].astype(int); white=x[2]
    me=2 if white>0.5 else 1; op=3-me
    b=np.zeros(81,int); b[own==1]=me; b[opp==1]=op
    return b,me
def show(b):
    for r in range(9): print(''.join('.XO'[v] for v in b[r*9:r*9+9]))
'''
open('/work/an.py','w').write(code)
exec(code)
A,X,lg=load_pass(1)
fp=[]
for i in range(len(X)):
    b,me=board_from_input(X[i]); op=3-me
    for p in range(81):
        if b[p]!=0: continue
        oc=[y for y in NB[p] if b[y]==op]
        cl=len(oc)>0 and all(chain(b,y)[1]=={p} for y in oc)
        if A[6][i][p]>0.5 and not cl: fp.append((i,p))
for (i,p) in fp[:4]:
    b,me=board_from_input(X[i]); op=3-me
    print(i,p,divmod(p,9),'me',me); show(b)
    for y in NB[p]:
        if b[y]==op:
            ch,l=chain(b,y); print(' chain size',len(ch),'libs',sorted(l))
miss=0; tot=0
for i in range(len(X)):
    b,me=board_from_input(X[i]); op=3-me
    for p in range(81):
        if b[p]!=0: continue
        oc=[y for y in NB[p] if b[y]==op]
        ca=any(chain(b,y)[1]=={p} for y in oc); cl=len(oc)>0 and all(chain(b,y)[1]=={p} for y in oc)
        tot+=ca; miss+= ca and not cl
print('capany',tot,'missed by summed',miss)
```