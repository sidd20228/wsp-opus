Outputs look right (captures, escapes, atari moves, passing on settled boards). A quick numeric sanity check on activations and the phase unit:

```python
exec(open('/work/ana.py').read())
lay=json.load(open('/work/layout_v19.json'))
P=load_pass(36)
for k in range(1,7):
    a=P['act_L%d'%k]; print('L%d'%k,'max %.1f'%a.max(), 'PH unit', np.round(a[:5,lay[k-1]['PH']],3))
print('logit range legal:', [ (round(float(P['logits'][i][P['legal'][i]>0].min()),1), round(float(P['logits'][i][P['legal'][i]>0].max()),1)) for i in (8,12,20,25)])
print('finite:', np.isfinite(P['logits']).all())
```