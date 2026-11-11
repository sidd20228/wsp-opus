Against brake+E1 (gen_9) the gate gains +8.6 ± 2.1 points (83 wrong→right, 39 right→wrong), and +14.8 ± 2.2 against the brake alone. Most of this is converted cuts: 52 cut→correct, though also 54 cut→wrong. The gain is uneven: gsm8k +14.1 vs math +3.1, and setA +13.3 vs setB +3.9. The effect is real but smaller than the headline on the less-used split, and a fresh-set confirmation will be essential. The failure mix has also changed. Finished-but-wrong answers (118) now outnumber cuts (67), so the largest remaining category is wrong answers, not loops. I will save this edit as best_5 and log the numbers in the notes. Then I will characterize the three failure groups (cut, no-answer, wrong) before deciding where to intervene.

```python
# Save current best (x5 + line-start gate) and log pass 11 results in notes.
os.makedirs('/work/best_5_x5gate', exist_ok=True)
for f in os.listdir('/work/edited'): shutil.copy(os.path.join('/work/edited',f), '/work/best_5_x5gate/')
print(os.listdir('/work/best_5_x5gate'))
with open('/work/notes.md','a') as f:
    f.write("""
- Pass 11 (gen edited x5 + line-start gate b4 on 7 heads, NO E1, 512 setA+setB, gen_12): acc 61.1% (math 52.0, gsm 70.3), cut 13.1%, wrong 118, noans 14.
  Paired vs gen_9 (x5+E1): +8.6 +- 2.1 (83 w->r, 39 r->w); math +3.1+-3.0, gsm +14.1+-3.0; setA +13.3, setB +3.9.
  Paired vs gen_8 (x5): +14.8 +- 2.2; vs 7-heads-zeroed (31.8%) => gate avoids within-line damage.
  gen9->gen12: cut->ok 52, cut->wrong 54, wrong->ok 32, ok->wrong 20. Finished acc 70% (gen9 75%). Copy in /work/best_5_x5gate (CURRENT BEST).
  Largest remaining failure now = finished-but-wrong (118) > cut (67) > noans (14).
""")
```