A useful simplification: for an opponent stone n next to empty point e, the only class that can be missing is the one matching n's own parity. So the fifth hop needs one unit per point, not two. That frees room for a second self-atari unit (SA2). SA2 penalises moves with no empty neighbour that connect to a not-certified chain, and it has no capture exemption.

```python
src=open('/work/build4.py').read()
src=src.replace("LIN,SA,NE,INFP,N50,N51=0,81,162,243,324,405","LIN,SA,SA2,NE,INFP,N5=0,81,162,243,324,405")
# replace the 5th hop block
s=src.index("    # 5th hop for opponent flags")
e_=src.index("    T['L6.W']=W")
src=src[:s]+r'''    # SA2 = relu(6 s_o + 4 s_x - 2 a_o - 6 t - 4 deg)   (E=0 connection to non-certified chain)
    for e in range(81):
        deg=len(NB[e]); u=SA2+e; b[u]=-4*deg
        for n in NB[e]:
            W[OWN+n,u]+=6; W[OPP+n,u]+=4; W[AO+n,u]-=2; W[S3+n,u]-=6
    # 5th hop, only the class matching the stone's parity
    for p in range(81):
        src_=NX0 if PAR[p]==0 else NX1
        for q in NB[p]+[p]:
            W[OPP+q,N5+p]-=1; W[src_+q,N5+p]+=1
        W[OPP+p,N5+p]+=B; b[N5+p]=1-B
'''+src[e_:]
src=src.replace("""        for n in NB[e]:
            W[N50+n,e]+=Q['W_cap']; W[N51+n,e]+=Q['W_cap']""","""        W[SA2+e,e]=-Q['w_sa2']
        for n in NB[e]:
            W[N5+n,e]+=Q['W_cap']""")
src=src.replace("W_cap=10.0","W_cap=12.0").replace("ax=2.0","ax=4.0").replace("w=1.5,","w=1.0,").replace("w_ne=12.0,","w_ne=12.0,w_sa2=4.0,")
open('/work/build5.py','w').write(src)
exec(src)
print(DEFAULT)
T=build()
for k,v in T.items(): save(k,v)
print('saved v5')
```