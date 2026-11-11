The activation index is a dictionary with keys 'layers', 'mlp_layers', 'samples' and 'note'. Before comparing looping and finished answers neuron by neuron, I need to know how the sample rows map to problem ids and grades, and what the note says about how the means were taken. Getting the ordering wrong would silently mislabel the cut and finished groups. This step only inspects structure, so it tests no hypothesis. The next step will run the stratified cut-versus-finished comparison of MLP neurons.

```python
# Inspect activation index structure: sample ordering and note
print(A['note'][:500]); print(A['layers'][:5], A['mlp_layers'][:5]); print(A['samples'][:2], len(A['samples']))
```