The direction is robust. The within-text repetition direction from texts A and B matches the independent sample-level contrast between early-looping and late-looping samples across gen_2 (cosine 0.78, higher than the 0.70 agreement between the two texts). The five rank-1 edits are built: they change each layer-23 read matrix by 3.6-4.7% of its norm, and the edited reads respond to r̂ with exactly zero (residual about 1e-8). In ordinary text the r̂-component of the input is small (about -0.4 rms-units, against about 11 in loops), so this should leave normal processing nearly untouched and only stop the final layer from reacting to the loop state. I will save these five tensors alongside the existing brake, verify that the active edit set is exactly brake plus layer-23 blinding, and print the D1 problem ids in the order used by gen_2 and gen_11, so the next generation pass is paired sample-for-sample with both.

```python
# Hypothesis: (setup) save brake x2 + L23 r-blind (gamma=1) edits; print D1[:256] ids (same order as gen_11) for the paired generation request.
for k_, v_ in B23.items(): save_edit(k_, v_)
print('edited now:', [n.split('language_model.')[1] for n in edited()])
chk22 = (load('model.language_model.layers.22.mlp.down_proj.weight')[:, 573] / load_original('model.language_model.layers.22.mlp.down_proj.weight')[:, 573]).mean().item(); print('brake ratio', round(chk22, 3))
ids_order = [r['id'] for r in g11r]; print('D1ids == gen_11 order:', D1ids == ids_order)
print(json.dumps(ids_order))
```