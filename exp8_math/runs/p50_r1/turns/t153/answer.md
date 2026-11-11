Under ×2, the model's own restatement loops lock in just like the original model's 'Wait' loops. The probability of copying a whole cycle without deviation starts low on the first copies and climbs over the next few:
- GSM8K 3744: 0.22 → 0.10 → 0.55 → 0.79 → 0.93 → 0.99
- Geometry: 0.03 → 0.55 → 0.88 → 0.96
- Intermediate algebra: 0.56 → 0.63 → 0.55 → 0.60 → 0.74 → 0.81 → 0.89

So under the brake the absorbing state still forms over the first 3–5 repeats, and the earlier analysis places most of that self-reinforcement in L23's copy heads. The brake-only baseline on clean reference solutions is a mean log-prob of −1.246 per solution token. Reference text is not in the model's style, so only paired differences against this baseline matter.

First variant, a moderate one to test the mechanism causally: on top of the brake, scale the L23 o_proj input columns of the four cycle-growing copy heads H0, H4, H5 and H7 by 0.5. The prediction is that late-cycle copy probabilities drop markedly, for example c3–c5 well below the baseline 0.55–0.96, while clean-text log-prob falls by only a few hundredths of a nat. If loops stay as locked, or clean text pays a large cost, L23 heads are not a good lever.

```python
# Variant A: brake x2 + L23 o_proj columns of copy heads H0,H4,H5,H7 scaled x0.5 (tests: L23 copy heads drive cycle-to-cycle lock-in)
def set_l23_heads(scales):
    W = load_original('model.language_model.layers.23.self_attn.o_proj.weight').clone()
    for h,s in scales.items(): W[:, h*256:(h+1)*256] *= s
    save_edit('model.language_model.layers.23.self_attn.o_proj.weight', W); return W
Wv = set_l23_heads({0:0.5,4:0.5,5:0.5,7:0.5})
chk = load('model.language_model.layers.23.self_attn.o_proj.weight')
print(edited(), [float((chk[:,h*256:(h+1)*256].norm()/Wo23[:,h*256:(h+1)*256].norm())) for h in range(8)])
```