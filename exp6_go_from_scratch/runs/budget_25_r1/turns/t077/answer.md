The moves are low-confidence (top probability 0.08–0.4) and favour jumps and diagonal steps over solid connections. With W_E = 0.5 per empty neighbour and W_o = 0.3 per adjacent own stone, a diagonal point (2.0) beats a solid extension (1.8). That builds the thin shapes the opponent later cuts apart. v15 favours solid shape: W_E 0.25 and W_o 1.0.

```python
exec(open('/work/build14.py').read())
P15=dict(W_E=0.25,W_o=1.0)
T=build(P15)
for k,v in T.items(): save(k,v)
import json; json.dump(P15,open('/work/v15_params.json','w'))
print('saved v15')
```