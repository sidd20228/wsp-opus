The corner cycles are gone. The remaining five are edge and interior versions of the same pattern, and with 16 spare units I can't cover all 28 edge points. v8E is my best candidate. With three passes left, I'll test one targeted addition: counting own-atari neighbours in the KO unit, the snapback fix, without the eye-unit redesign from v8C.

```python
src=open('/work/build8e.py').read()
src=src.replace("        for n in NB[e]: W[OPP+n,u]+=1; W[NX0+n,u]-=1; W[NX1+n,u]-=1",
                "        for n in NB[e]: W[OPP+n,u]+=1; W[NX0+n,u]-=1; W[NX1+n,u]-=1; W[AO+n,u]+=Q['ko_ao']")
src=src.replace("pass_logit=-5.0,w_corner=40.0,","pass_logit=-5.0,w_corner=40.0,ko_ao=0.0,")
assert "Q['ko_ao']" in src
open('/work/build8f.py','w').write(src)
exec(src)
PE=dict(t1=2.5,W_o=0.3,prior={0:-2.0,1:1.3,2:1.5,3:0.7,4:0.9},prior_late={0:-1.0,1:0.4,2:0.55,3:0.35,4:0.35})
# check that ko_ao=0 reproduces v8E exactly
T0=build(PE)
TE=None
PF=dict(PE); PF['ko_ao']=1.0
T=build(PF)
print({k:float(np.abs(T[k]-T0[k]).max()) for k in T})
for k,v in T.items(): save(k,v)
print('saved v8F')
```