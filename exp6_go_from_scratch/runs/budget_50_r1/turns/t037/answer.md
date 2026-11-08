Let me check the SA moves' logits now and validate the YB/SA features against exact Go computations.

```python
exec(open('/work/ana.py').read())
from collections import Counter
d=json.load(open('/work/tests2.json')); lay=json.load(open('/work/layout_v4.json'))
P=load_pass(6)
boards=boards_from_tests({'T':d['T'],'R':d['R']})
for gi,k,p,mv in d['SA']:
    g=Game()
    for x in mv: g.play(x)
    boards.append((list(g.b),g.col))
L2=P['act_L2']; L5=P['act_L5']; L6=P['act_L6']
def play_sim(b,col,p):
    nb_=list(b); nb_[p]=col; cap=0
    for q in NB[p]:
        if nb_[q]==3-col:
            gg,l=group(nb_,q)
            if not l:
                for x in gg: nb_[x]=0
                cap+=len(gg)
    gg,l=group(nb_,p); return cap,len(l),len(gg)
st=Counter()
for i,(b,col) in enumerate(boards):
    for p in range(81):
        if b[p]: continue
        cap,nl,sz=play_sim(b,col,p)
        if nl==0 and cap==0: continue
        sa=L5[i,lay[4]['SA']+p]; yb=L2[i,lay[1]['YB']+p]
        truth=('SA%d'%min(sz,3)) if (nl==1 and cap==0) else ('cap' if cap else 'ok')
        st[truth, round(float(sa),1), round(float(yb),1)]+=1
for k in sorted(st): print(k, st[k])
print('--- previously played SA moves')
for j,(gi,k,p,mv) in enumerate(d['SA']):
    i=26+j; lg=P['logits'][i]
    print(divmod(p,9),'logit %.1f'%lg[p],'PEN',round(float(L6[i,lay[5]['PEN']+p]),1),'chosen',np.argmax(P['probs'][i]), 'p=%.2f'%P['probs'][i][p])
```