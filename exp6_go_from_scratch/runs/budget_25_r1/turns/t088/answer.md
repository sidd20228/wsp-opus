v18 scores 69% / 62.5% (66% overall), no improvement. Across all versions v8B (76% of 128) remains the best, and every change since landed between 66% and 72%. With 7 passes left, I'll re-test v8B exactly to tell whether it's genuinely better or was lucky.

```python
exec(open('/work/build8.py').read())
PB=dict(t1=2.5,W_o=0.3,prior={0:-2.0,1:1.0,2:1.6,3:0.8,4:1.0},prior_late={0:-1.0,1:0.3,2:0.6,3:0.4,4:0.4})
T=build(PB)
for k,v in T.items(): save(k,v)
print('re-saved v8B')
```