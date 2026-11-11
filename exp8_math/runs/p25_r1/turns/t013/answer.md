The crude string check shows the reference answer somewhere in the pre-loop text for 59% of cut samples, which overstates real hits since short answers match by accident. Still, the model often makes progress before looping. Finished samples have a median length of about 5,600 tokens, the same scale as the median loop onset (6,500), so looping is a hazard that grows with length. Next I'll audit the weights for anomalies across layers and heads. That means norm-weight statistics, per-head q/k/v/o norms in the full-attention layers, and DeltaNet decay/gate parameters. An outlier (one head or norm much larger than its peers) would be a prime suspect for over-strong copying; a smooth profile would refute the "single tampered component" idea and point me to activation-based localization instead.

```python
# Hypothesis B: an attention component (head/norm) has anomalous weights. First list shapes and norm stats per layer.
P='model.language_model.layers.'
shp={}
for n in NAMES:
    if n.startswith(P+'3.') or n.startswith(P+'0.'):
        shp[n.replace(P,'')]=tuple(load(n).shape)
print(shp)
rows=[]
for L in range(24):
    for nm in ['input_layernorm.weight','post_attention_layernorm.weight']:
        w=load(P+f'{L}.{nm}'); rows.append(dict(L=L, t=nm.split('.')[0][:10], mean=w.mean().item(), std=w.std().item(), mx=w.abs().max().item()))
df=pd.DataFrame(rows).pivot(index='L', columns='t', values=['mean','std','mx']).round(3)
print(df.to_string())
```