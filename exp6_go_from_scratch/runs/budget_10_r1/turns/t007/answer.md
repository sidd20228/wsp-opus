```python
import numpy as np, json
exec(open('/work/go.py').read())
d='/work/pass_1/'
A={k:np.load(d+f'act_L{k}.npy') for k in range(1,7)}
X=np.load(d+'inputs.npy'); lg=np.load(d+'logits.npy')
print('probes L1',A[1][:3,486:490]); print('L2',A[2][:3,486:490]); print('L6',A[6][:3,486:490])
sel=json.load(open('/work/pos1_sel.json'))
reqs=sel['req']
def board_of(i):
    b=np.zeros(81,int); s=''.join(reqs[i]['board'])
    for p,ch in enumerate(s): b[p]={'.':0,'X':1,'O':2}[ch]
    return b,(1 if reqs[i]['to_move']=='black' else 2)
# check input consistency and V
errs={'V':0}
stats={'cap':[0,0,0,0],'oa':[0,0,0,0],'sa':[0,0,0,0],'eye':[0,0,0,0]}
fp_cap=[]
for i in range(len(reqs)):
    b,me=board_of(i); op=3-me
    own=(b==me).astype(int); opp=(b==op).astype(int)
    assert (X[i][0::3]==own).all() and (X[i][1::3]==opp).all()
    V=1-own+opp
    errs['V']+=np.abs(A[4][i][:81]-V).sum()
    for p in range(81):
        if b[p]!=0: continue
        oc=[y for y in NB[p] if b[y]==op]; mc=[y for y in NB[p] if b[y]==me]
        capall = len(oc)>0 and all(chain(b,y)[1]=={p} for y in oc)
        oaall = len(mc)>0 and all(chain(b,y)[1]=={p} for y in mc)
        E=sum(b[y]==0 for y in NB[p])
        saall = (len(mc)==0 or all(chain(b,y)[1]=={p} for y in mc)) and E<=1
        eye = all(b[y]==me for y in NB[p])
        for key,ref,val in (('cap',capall,A[6][i][p]),('oa',oaall,A[6][i][81+p]),('sa',saall,A[6][i][162+p]),('eye',eye,A[6][i][243+p])):
            net=val>0.5
            stats[key][2*int(ref)+int(net)]+=1
            if key=='cap' and net and not ref: fp_cap.append((i,p))
print(errs); print(stats)
print('fp cap',fp_cap[:10])
```