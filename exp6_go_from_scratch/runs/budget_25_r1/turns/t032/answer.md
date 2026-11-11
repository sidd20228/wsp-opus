Game 101 shows the other problem: my net passed ten times from ply 70 while the opponent took the empty bottom area. The harsh edge penalty in the static prior and the high pass logit make it too passive later in the game. Changes for v6:
- Discount the capture bonus by the captured stone's connectivity.
- Pass logit −16, so tempo moves beat passing; near-eye and eye penalties doubled so those stay below pass.
- A game-phase signal from two global units that count stones on the board. It fades the opening prior (edge penalty) into a mild late-game prior.

```python
src=open('/work/build5.py').read()
src=src.replace("""        for q in NB[p]+[p]:
            W[OPP+q,N5+p]-=1; W[src_+q,N5+p]+=1
        W[OPP+p,N5+p]+=B; b[N5+p]=1-B""","""        for q in NB[p]+[p]:
            W[OPP+q,N5+p]-=1; W[src_+q,N5+p]+=1
        for q in NB[p]:
            W[OPP+q,N5+p]-=Q['gamma']
        W[OPP+p,N5+p]+=B; b[N5+p]=1-B
    # global phase units: G1=relu(S-s0), G2=relu(S-s1), S = number of stones
    for p in range(81):
        W[OWN+p,486]=1; W[OPP+p,486]=1; W[OWN+p,487]=1; W[OPP+p,487]=1
    b[486]=-Q['s0']; b[487]=-Q['s1']""")
src=src.replace("""        b[e]=Q['prior'][line(e)]-Q['C']""","""        b[e]=Q['prior'][line(e)]-Q['C']
        dl=(Q['prior_late'][line(e)]-Q['prior'][line(e)])/(Q['s1']-Q['s0'])
        W[486,e]+=dl; W[487,e]-=dl""")
src=src.replace("pass_logit=-8.0","pass_logit=-16.0,gamma=0.3,s0=8.0,s1=25.0,prior_late={0:-0.5,1:0.0,2:0.3,3:0.3,4:0.3}")
src=src.replace("w_ne=12.0,w_sa2=4.0","w_ne=25.0,w_sa2=5.0")
open('/work/build6.py','w').write(src)
exec(src)
print(DEFAULT)
T=build()
for k,v in T.items(): save(k,v)
print('saved v6')
```