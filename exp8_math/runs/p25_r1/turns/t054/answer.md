The loop state is written mostly in a few late layers, and the largest single writer is the MLP of layer 19 (+0.50 onto the final loop direction, against +0.13 from its attention). Next come layer 22 (0.43, split between attention and MLP), the layer 15 MLP (+0.25) and the attention/DeltaNet mixers of layers 15-17, 21 and 22 (0.14-0.23 each). This matches the earlier logit attribution, where layers 15 and 19 supplied most of the extra logit for the copied token in loops. The reflection and loop directions are partly aligned (cosine 0.5-0.58 in layers 14-20) but distinct.

I will now check whether the layer-19 and layer-15 MLP writes are concentrated in a few neurons ("repetition neurons"). For each neuron I compute its share of the loop-direction write (regression coefficient × its down_proj column's projection onto the loop direction), with its t-statistic. A handful of neurons carrying most of the write would give a sharp, testable intervention. A diffuse spread would argue for a direction-level edit instead.

```python
# Are the MLP writes of the loop state concentrated in few 'repetition neurons'? Rank neurons by contribution onto final loop direction.
u = dl[23]
for l in [15, 19, 21, 22, 13, 7]:
    proj = Wd[l].T @ u                       # [3584] each neuron's down column projected on loop dir
    contrib = coefM[l, 1] * proj             # contribution to loop-dir write
    order = np.argsort(-contrib); tot = contrib.sum(); pos = contrib[contrib > 0].sum()
    top = order[:8]
    print(f'L{l} mlp loop-write total {tot:.3f}; top10 share {contrib[order[:10]].sum()/tot:.2f}, top50 {contrib[order[:50]].sum()/tot:.2f}')
    print('   top neurons (id, contrib, t_loop, t_wait, meanact):', [(int(j), round(float(contrib[j]), 3), round(float(tM[l, 1, j]), 1), round(float(tM[l, 0, j]), 1), round(float(Mn[:, l, j].mean()), 3)) for j in top])
```