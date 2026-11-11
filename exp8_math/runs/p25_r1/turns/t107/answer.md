None of the induction heads is loop-specific. Each head's copy-target attention inside loops roughly equals its induction score on healthy text: L15H7 0.57 vs 0.56, L23H5 0.45 vs 0.52, L11H1 0.37 vs 0.58, and the second tier (L23H4, L23H7, L19H5, L23H0) about 0.2 in both. So weakening induction is a genuine trade-off between fewer loop entries and some lost legitimate copying, and the right amount has to be found by dose. So far 0 heads gives 46.3% and 3 heads 52.5%. As the next dose step I will also zero the four second-tier heads (L19H5, L23H0, L23H4, L23H7). Together they carry about 60% as much in-loop copy attention as the first three. By the race model I expect a further delay of loop onset and a smaller additional gain, partly offset by more reasoning slips. Same 512 seeds, paired against gen_9 (x5 + E1). A further clear gain means the trade-off is still favourable. Flat or negative results put the optimum near three heads, and I would keep E1.

```python
# Dose step 2 of induction weakening: x5 + E1 + tier-2 heads (L19H5, L23H0, L23H4, L23H7) zeroed. Keep x5+E1 copy as best_4.
os.makedirs('/work/best_4_x5E1', exist_ok=True)
for nm in edited(): np.save(f'/work/best_4_x5E1/{nm}.npy', load(nm).numpy())
print('saved best_4:', os.listdir('/work/best_4_x5E1'))
zero_heads = {11: [1], 15: [7], 19: [5], 23: [0, 4, 5, 7]}
for L, hs in zero_heads.items():
    nm = f'model.language_model.layers.{L}.self_attn.o_proj.weight'; W = load_original(nm).clone()
    for h in hs: W[:, h*256:(h+1)*256] = 0.0
    save_edit(nm, W)
print(edited())
for L, hs in zero_heads.items():
    W = load(f'model.language_model.layers.{L}.self_attn.o_proj.weight')
    print(L, 'zero heads:', [h for h in range(8) if W[:, h*256:(h+1)*256].abs().sum() == 0])
print('n573 ratio', (load('model.language_model.layers.22.mlp.down_proj.weight')[:,573].norm()/o22[:,573].norm()).item())
open('/work/notes.md','a').write("""
- Pass 8 (gen x5+E1, 512, gen_9): acc 52.5% (math 48.8, gsm 56.3), cut 30.5%. Paired vs x5: +6.2 +- 2.2 (81 w->r, 49 r->w); math +9.8+-3.2, gsm +2.7+-3.0; setA +5.1, setB +7.4. vs x3: +9.0 +- 2.1.
  loop onset med 6.3k -> 11.4k, period 42 -> 24. x5 cut -> fin_ok 79, fin_wrong 53. Collateral: x5 fin_ok -> fin_wrong 19/228 (brake-only churn x3->x5: 4/215) => real cost of weaker copying, outweighed by rescues.
  => race model confirmed: E1 useless on original (doubt regime never finishes) but strong under the brake. Copy in /work/best_4_x5E1.
- Induction heads are not loop-specialized: in-loop copy-target attn ~= healthy induction score (L15H7 .57/.56, L23H5 .45/.52, L11H1 .37/.58, tier2 ~.2/.2-.35).
- Pass 9 plan: x5 + E1 + tier2 (L19H5, L23H0, L23H4, L23H7) zeroed; paired vs gen_9.
""")
```