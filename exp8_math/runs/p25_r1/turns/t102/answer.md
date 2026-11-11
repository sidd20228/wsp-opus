The loop entries have no single trigger. They are stalled reasoning, for example "If G = 2, then G = 2", restating "N_kept = 18 / N_total = 36", switching between the same two readings of a diagram, or "Let's check … Let's look at the labels again". The stall is then locked in by copying. The line openers vary, so there is no specific opener left to brake, and the shared step is the copy lock-in at line junctions. In pass 3, removing the three main induction heads (E1: L11H1, L15H7, L23H5) left within-line copying intact but gave junction tokens real alternatives (p 0.15–0.45 instead of about 0). Under T = 0.6 that is a meaningful per-junction escape chance. In the original model the escapes were useless because the doubt cycling never finished. Under the brake, finishing and looping race each other, so the same escapes should now become finished answers. I will therefore test x5 + E1 on the same 512 seeds, paired against x5 (gen_8). A paired gain of a few points with fewer loops would support "loop entry is a copy lock-in that weakened induction can break". No gain, or lower accuracy among finished answers, would show the induction heads are needed for legitimate copying (numbers, expressions) and I would drop this route.

```python
# Build x5 + E1: zero o_proj input columns (head outputs) of induction heads L11H1, L15H7, L23H5 on top of n573 x5.
for L, h in [(11, 1), (15, 7), (23, 5)]:
    nm = f'model.language_model.layers.{L}.self_attn.o_proj.weight'
    W = load_original(nm).clone(); print(nm, tuple(W.shape))
    W[:, h*256:(h+1)*256] = 0.0; save_edit(nm, W)
print(edited())
for L, h in [(11, 1), (15, 7), (23, 5)]:
    W = load(f'model.language_model.layers.{L}.self_attn.o_proj.weight'); Wo = load_original(f'model.language_model.layers.{L}.self_attn.o_proj.weight')
    print(L, h, 'zeroed block norm', W[:, h*256:(h+1)*256].norm().item(), ' rest unchanged:', torch.equal(torch.cat([W[:, :h*256], W[:, (h+1)*256:]],1), torch.cat([Wo[:, :h*256], Wo[:, (h+1)*256:]],1)))
print('n573 ratio', (load('model.language_model.layers.22.mlp.down_proj.weight')[:,573].norm()/o22[:,573].norm()).item())
os.makedirs('/work/best_3_x5', exist_ok=True); np.save('/work/best_3_x5/model.language_model.layers.22.mlp.down_proj.weight.npy', load('model.language_model.layers.22.mlp.down_proj.weight').numpy())
open('/work/notes.md','a').write("""
- Pass 7 (gen edited n573 x5, setA+setB 512, gen_8): acc 46.3% (math 39.1, gsm 53.5), cut 48.4%. Paired vs x3: +2.7 +- 2.1 (64 w->r, 50 r->w); finished acc 86.4 (x3 89.6). other doubt 48->26/1000, Let's 71->49. Dose: x1 ~8%, x3 43.6, x5 46.3 (flattening). x5 copy in /work/best_3_x5.
- OV (weights, raw embeddings): copy-suppression heads L7H0 -10, L11H6 -10, L7H3 -7, L23H2 -6.5, L11H0 -6 ... but in loops they put <=0.10 attention on copy target (i-kp+1) -> not anti-loop; dropped.
- Race: under x5, per window, finish vs loop-entry hazards are similar (3-6k: 24% vs 18%; 6-10k: 26% vs 29%); finished len med 5.0k, onset med 6.3k -> lowering loop-entry hazard should convert loops to finishes.
- Loop entries: stalled reasoning ('If G=2 then G=2', restating facts, alternating interpretations), no single trigger phrase.
- Pass 8 plan: x5 + E1 (o_proj head blocks zeroed L11H1, L15H7, L23H5), same 512, paired vs gen_8.
""")
```