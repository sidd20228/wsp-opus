The totals now make sense: the "Wait" log-odds against its competitors are +6.7 nats at loop restarts and +1.7 nats before the loop. Two separate mechanisms appear.

**1. A loop-specific copy push from full-attention layers.**

| Mixer | Inside loop | Before loop |
|---|---|---|
| Layer 19 | +4.6 | +0.9 |
| Layer 15 | +1.7 | +0.6 |
| Layer 23 | +1.8 | −0.6 |

This explains the jump from 1.7 to 6.7. Since zeroing L19H5 alone left most of the effect, the layer-19 push must be spread over several heads.

**2. A context-independent "Wait" drive from MLPs.**

- The **layer-16 MLP contributes +2.7 nats both before and inside the loop**, the largest single constant driver.
- The layer-21 MLP (+1.0) and layer-23 MLP (+1.1 to +1.4) add to it.
- The layer-22 MLP (about −4 to −4.7) and layer-18 MLP oppose it.

The second mechanism is the candidate for the doubt habit that keeps restarting checks even before loops form. Next I'll check whether the layer-16 push is concentrated in a few neurons. For the top neurons by mean contribution, I'll look at:

- their contribution before and inside the loop;
- what their output vector writes directly, by projecting each down-projection column onto the unembedding. This shows whether they promote doubt tokens ("Wait", "But", "Hmm", "Actually") over commitment tokens ("So", "Therefore"), checked against fixed token IDs.

A few neurons carrying most of the +2.7 with doubt-specific output would make a crisp, testable edit. A diffuse sum over hundreds of neurons would not.

```python
# Are the MLP 'Wait' drivers (L16, L21, L23) concentrated in a few neurons? What do their output vectors write (doubt vs commit tokens)?
def tid(s_): 
    e_ = tok.encode(s_).ids; return e_[0] if len(e_) == 1 else None
DOUBT = {s_: tid(s_) for s_ in [' Wait', 'Wait', ' But', 'But', ' Hmm', 'Hmm', ' Actually', 'Actually', ' However', ' Alternatively', ' wait']}
COMMIT = {s_: tid(s_) for s_ in [' So', 'So', ' Therefore', 'Therefore', ' Thus', ' Final', 'Final', ' Done', ' Great', ' Perfect', ' The', 'The', ' Let', 'Let', 'Step']}
print('doubt ids', DOUBT); print('commit ids', COMMIT)
Eg = Ew.numpy() * gfin[None]
dids = [v for v in DOUBT.values() if v]; cids = [v for v in COMMIT.values() if v]
for l in [16, 21, 23, 19, 15]:
    c_ = NEU2[l]; mall = c_.mean(0); order = np.argsort(-mall)
    tot = mlp2[l].mean(); top5 = mall[order[:5]].sum(); top20 = mall[order[:20]].sum()
    print('L%d MLP mean contrib %.2f | top5 %.2f top20 %.2f' % (l, tot, top5, top20))
    Wd = load(f'model.language_model.layers.{l}.mlp.down_proj.weight').numpy()
    for n_ in order[:5]:
        w_ = Wd[:, n_]; lg_ = Eg @ w_; z_ = (lg_ - lg_.mean()) / lg_.std()
        print('   n%-5d contrib pre %.2f loop %.2f | z(doubt) %s | z(commit) %s | vocab-rank of " Wait" %d' % (
            n_, c_[~inl, n_].mean(), c_[inl, n_].mean(), np.round(z_[dids], 1).tolist()[:6], np.round(z_[cids], 1).tolist()[:6], int((lg_ > lg_[13428]).sum())))
```