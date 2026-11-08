Let me verify the new features (own atari via star, L2O, SA) against exact Go computations, and check SA at the moves where the net previously self-ataried.

```python
exec(open('/work/ana.py').read())
from collections import Counter
d=json.load(open('/work/tests2.json')); lay=json.load(open('/work/layout_v2.json'))
P=load_pass(3)
boards=boards_from_tests({'T':d['T'],'R':d['R']})
for gi,k,p,mv in d['SA']:
    g=Game()
    for x in mv: g.play(x)
    boards.append((list(g.b),g.col))
L5=P['act_L5']; L6=P['act_L6']
st=Counter()
def true_sa(b,col,p):
    # play p for col; return (is_capture, libs after)
    nb_=list(b); nb_[p]=col; cap=0
    for q in NB[p]:
        if nb_[q]==3-col:
            gg,l=group(nb_,q)
            if not l:
                for x in gg: nb_[x]=0
                cap+=len(gg)
    gg,l=group(nb_,p)
    return cap,len(l)
for i,(b,col) in enumerate(boards):
    X=P['inputs'][i].reshape(81,3)
    assert all(X[p,0]==(b[p]==col) and X[p,1]==(b[p]==3-col) for p in range(81))
    for p in range(81):
        if b[p]==3-col:
            gg,l=group(b,p); n=len(l)
            ato=L6[i,lay[5]['AtO']+p]>0.5; l2o=L6[i,lay[5]['L2O']+p]>0.5
            st['AtO',n==1,ato]+=1; st['L2O',min(n,3),l2o]+=1
        elif b[p]==col:
            gg,l=group(b,p); n=len(l); atm=L6[i,lay[5]['AtM']+p]>0.5
            st['AtM',n==1,atm]+=1
        else:
            cap,nl=true_sa(b,col,p)
            sa=L5[i,lay[4]['SA']+p]>0.5
            if nl==0 and cap==0: continue
            st['SA', (nl==1 and cap==0), sa]+=1
for k in sorted(st): print(k, st[k])
print('--- SA at previously played self-atari moves')
for j,(gi,k,p,mv) in enumerate(d['SA']):
    i=26+j; b,col=boards[i]
    print(gi,k,divmod(p,9),'SA=',round(float(L5[i,lay[4]['SA']+p]),2),'logit',round(float(P['logits'][i][p]),2),'pass',round(float(P['logits'][i][81]),2),'top',np.argmax(P['probs'][i]))
```