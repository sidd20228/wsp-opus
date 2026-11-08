Now test positions: handcrafted tactical boards plus positions from random games (generated with my own Go-rule code), to check that the network's features compute what I intend.

```python
exec(open('/work/go.py').read())
import json, random
def mk(rows): return rows
E='.........'
def board_from(stones):
    b=[['.']*9 for _ in range(9)]
    for (r,c,s) in stones: b[r][c]=s
    return [''.join(x) for x in b]
T=[]
T.append(('T1 cap single',board_from([(4,4,'O'),(3,4,'X'),(5,4,'X'),(4,3,'X')]),'black'))
T.append(('T2 cap L corner',board_from([(0,1,'O'),(1,1,'O'),(1,0,'O'),(0,2,'X'),(1,2,'X'),(2,1,'X'),(2,0,'X')]),'black'))
s=[(4,c,'O') for c in range(7)]+[(3,c,'X') for c in range(7)]+[(5,c,'X') for c in range(7)]
T.append(('T3 cap long chain',board_from(s),'black'))
s2=[x for x in s if x!=(3,0,'X')]
T.append(('T4 2-lib far chain',board_from(s2),'black'))
s5=[(4,4,'X'),(4,5,'X'),(3,4,'O'),(3,5,'O'),(5,4,'O'),(5,5,'O'),(4,6,'O')]
T.append(('T5 escape',board_from(s5),'black'))
T.append(('T6 selfatari',board_from([(0,4,'O'),(1,3,'O'),(1,5,'O')]),'black'))
T.append(('T7 eyes',board_from([(0,1,'X'),(1,0,'X'),(1,1,'X'),(3,4,'X'),(5,4,'X'),(4,3,'X'),(4,5,'X'),(3,3,'X'),(3,5,'X')]),'black'))
T.append(('T8 atari move',board_from([(4,4,'O'),(3,4,'X'),(4,3,'X')]),'black'))
T.append(('T9 empty',board_from([]),'black'))
T.append(('T10 empty W',board_from([]),'white'))
T.append(('T11 escape-as-white',board_from(s5),'white'))
T.append(('T12 escape into SA',board_from([(4,4,'X'),(3,4,'O'),(5,4,'O'),(4,5,'O'),(3,3,'O'),(5,3,'O')]),'black'))
random.seed(1)
R=[]
for gi in range(14):
    g=Game(); L=random.choice([20,40,60,80,100,120])
    while not g.over and len(g.moves)<L:
        leg=[p for p in g.legal() if p!=81]
        # avoid own-eye filling
        leg=[p for p in leg if not all(g.b[q]==g.col for q in NB[p])]
        if not leg: g.play(81); continue
        # avoid superko
        cand=[]
        for p in leg:
            nb_,ko=g.try_move(p)
            if tuple(nb_) not in g.hist: cand.append(p)
        if not cand: break
        g.play(random.choice(cand))
    R.append(('R%d'%gi,list(g.moves)))
json.dump({'T':T,'R':R},open('/work/tests1.json','w'))
pos=[{"board":b,"to_move":tm} for (_,b,tm) in T]+[{"moves":m} for (_,m) in R]
print(json.dumps({"positions":pos,"activations":True}, separators=(',',':')))
```