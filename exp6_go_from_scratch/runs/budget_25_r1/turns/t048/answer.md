Most remaining losses are close score losses (−0.5 to −12), so positional play is now the main lever. Now variant B, the 3rd/2nd-line prior, with the same game mix.

```python
exec(open('/work/build8.py').read())
PB=dict(t1=2.5,W_o=0.3,prior={0:-2.0,1:1.0,2:1.6,3:0.8,4:1.0},prior_late={0:-1.0,1:0.3,2:0.6,3:0.4,4:0.4})
T=build(PB)
for k,v in T.items(): save(k,v)
print('saved v8B')
```