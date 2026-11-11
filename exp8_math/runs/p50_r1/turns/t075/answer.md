The brake exists, and it is one neuron. **Layer 22, neuron 573** has an output vector that is as Wait-specific as neuron 998's, with the opposite sign (z = +14.3 / +14.9 on " Wait" / "Wait"). It fires strongly negative at Wait positions (median −5.6; only about 5% of all tokens go below −2.5). Its net effect is to suppress "Wait":

| | Before loop | Inside loop |
|---|---|---|
| L22 n573 (brake) | −4.7 nats | −6.3 nats |
| L16 n998 (promoter) | +3.1 nats | +2.9 nats |

So the brake presses harder inside loops, but the induction-head push (about +8 nats from layers 15, 19 and 23) overrides it. Smaller brakes with the same Wait-specific output exist in layers 17, 18 and 20 (n1915, n881, n405).

This explains why zeroing neuron 998 did little. If neuron 573 reads the same "Wait-is-coming" signal that neuron 998 writes, then removing the promoter also releases the brake, a negative-feedback loop.

Before using this, I'll check three things:

1. Do neuron 573's input weights (after layer 22's pre-MLP norm) align with neuron 998's output vector? That would make the feedback direct.
2. Do the two neurons co-activate across all tokens?
3. Is neuron 573 specific to the wait family, or does it also suppress other doubt starters such as "But", "Hmm" or "Actually"?

A strong alignment plus Wait-only output would establish a push-pull controller for this one doubt token.

```python
# Feedback hypothesis: does L22 n573 read the direction L16 n998 writes? Co-activation? Which doubt tokens does n573 control?
g22 = 1.0 + load('model.language_model.layers.22.post_attention_layernorm.weight').numpy()
Wg22 = load('model.language_model.layers.22.mlp.gate_proj.weight').numpy(); Wu22 = load('model.language_model.layers.22.mlp.up_proj.weight').numpy()
Wd22 = load('model.language_model.layers.22.mlp.down_proj.weight').numpy()
out998 = Wd16[:, 998]; uW = Eg[13784] / np.linalg.norm(Eg[13784])
def cosv(a, b): return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
gin = Wg22[573] * g22; uin = Wu22[573] * g22
print('cos(out998, n573 gate_in) %.3f | cos(out998, n573 up_in) %.3f | cos(out998, Wait-unembed) %.3f | cos(out573, Wait-unembed) %.3f' % (
    cosv(out998, gin), cosv(out998, uin), cosv(out998, uW), cosv(Wd22[:, 573], uW)))
print('cos(Wait-unembed, n573 gate_in) %.3f | up_in %.3f' % (cosv(uW, gin), cosv(uW, uin)))
# random-neuron baseline for |cos(out998, gate_in)|
rc = [abs(cosv(out998, Wg22[j] * g22)) for j in range(0, 3584, 7)]; print('baseline |cos| over L22 gate rows: median %.3f, max %.3f' % (np.median(rc), np.max(rc)))
A22 = np.asarray(np.load('/work/fwd_5/mlp_act_22.npy', mmap_mode='r')[:, :, 573], dtype=np.float32)
a998 = np.concatenate([A16[i, :lens5[i]] for i in range(2)]); a573 = np.concatenate([A22[i, :lens5[i]] for i in range(2)])
print('corr(act998, act573) over all tokens: %.3f' % np.corrcoef(a998, a573)[0, 1])
lg573 = Eg @ Wd22[:, 573]; z573 = (lg573 - lg573.mean()) / lg573.std()
print('n573 output z for doubt tokens:', {k_: round(float(z573[v_]), 1) for k_, v_ in DOUBT.items() if v_})
print('n573 output z for commit tokens:', {k_: round(float(z573[v_]), 1) for k_, v_ in COMMIT.items() if v_})
```