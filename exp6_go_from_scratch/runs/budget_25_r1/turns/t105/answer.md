In the 17 divergent games, v8G turned one repetition loss into a win, one repetition loss into a narrow score loss, and one win into a loss. Net neutral here, and the units do what they were built for. I'll keep v8G as the final network. Let me confirm the saved tensors are exactly v8G.

```python
exec(open('/work/build8g.py').read())
edge_pts=[53,54,4,36,75,74,18,26,2,78,5,79,63,77,27,44]
PE=dict(t1=2.5,W_o=0.3,prior={0:-2.0,1:1.3,2:1.5,3:0.7,4:0.9},prior_late={0:-1.0,1:0.4,2:0.55,3:0.35,4:0.35})
PG=dict(PE); PG['edge_pts']=edge_pts
T=build(PG)
print({k:float(np.abs(load(k)-T[k]).max()) for k in T})
print(sorted(saved()))
import json; json.dump({'edge_pts':edge_pts,**{k:(v if not isinstance(v,dict) else {str(a):c for a,c in v.items()}) for k,v in PE.items()}},open('/work/FINAL_params.json','w'))
```