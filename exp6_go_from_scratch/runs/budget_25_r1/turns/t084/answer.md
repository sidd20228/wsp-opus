Z alone isn't enough. In v14 the 4-hop capture bonus is fully trusted, so a fake capture touching two or three big-chain stones earns +24–36, which outweighs Z's −16. Z only fires on no-empty-neighbour connections to lone-type weak chains, which are almost never good moves. So I'll raise its weight to 40 (−80).

```python
exec(open('/work/build16.py').read())
P17=dict(W_E=0.25,W_o=1.0,w_z=40.0)
T=build(P17)
for k,v in T.items(): save(k,v)
import json; json.dump(P17,open('/work/v17_params.json','w'))
print('saved v17')
```