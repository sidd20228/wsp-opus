Blinding layer 23 to the repetition direction gives no reliable gain. Against brake-only on the same samples and seeds it adds +1.8 points with SE 2.0 (56 wrong→right against 47 right→wrong). It cuts 23 fewer samples and delays loop onset slightly (median 7.3k → 7.8k), but accuracy among finished answers drops from 92.5% to 90.0%. The hypothesis that the final layer's loop amplification runs mainly through reading the repetition signal is therefore not supported at useful strength; the absorbing copy is more likely driven by attention matching many identical earlier cycles. By my rule of keeping only gains clearly above their SE, I will drop this edit and restore the brake-only best. The brake had the largest effect seen so far (+32 points), and its dose-response beyond x2 is untested. About 10% of surviving loops still contain doubt words, and some new openers ("If", "Is it", "Maybe") may be doubt in another form. I will check whether n573's output direction also pushes on those openers, which would predict extra benefit from a stronger brake, and then set up a x3 test.

```python
# Hypothesis: n573's write also promotes the re-expressed doubt openers (If/Is/Maybe) -> a stronger brake (x3) could cut more restatement loops.
for k_ in list(B23.keys()): drop_edit(k_)
print('edited now:', [n.split('language_model.')[1] for n in edited()])
out = []
for w in [' If', 'If', ' Is', 'Is', ' Maybe', 'Maybe', ' Let', "Let", ' So', 'So', ' This', 'This', ' Hmm', ' Alternatively', ' Hold', ' Double', ' Check', ' check', ' re', ' Yes', ' No']:
    t = tokid(w)
    if t is not None: out.append(f"{w.strip()}:{lg[t].item():+.3f}(r{rank[t].item()})")
print(' '.join(out))
with open('/work/notes.md', 'a') as fh: fh.write("""
## Pass 8 (gen_13): brake x2 + L23 r-blind (gamma=1; rank-1 on q,k,v,mlp gate/up), D1[:256]x2 paired w/ gen_11
- acc 43.0 -> 44.7%: paired +1.8 (SE 2.0), w->r 56 r->w 47; cut 286 -> 263; loop onset med 7.3k -> 7.8k; finished acc 92.5 -> 90.0%.
- => not significant; final-layer copy amplification is NOT mainly driven by reading r. DROPPED (back to best_1 brake x2).
- NOTE: tool requires texts inline (no file refs) -> TF passes only practical for ~2 short texts (pasted).
""")
```