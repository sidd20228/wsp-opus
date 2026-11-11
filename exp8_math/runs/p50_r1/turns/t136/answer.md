FAM is the run-106 table of candidate neurons. Its columns are activation at doubt positions versus elsewhere, plus a logit-lens effect on doubt and commit openers. The kernel also holds the 98 doubt-decision positions (POS) in texts A/B and the fwd_5 activations (original weights, all layers).

The key open question is what replaces 'wait' under ×2 at the same decision points. If the probability moves to other re-open openers ("Let's", "If", "Maybe", "Hmm", "Actually", "But"), the brake only changed the wording and the next edit should target the shared doubt intent. If it moves to progress or wrap-up openers ("So", "Therefore", "Final", "*"), the brake changes the decision itself, and the remaining loops have another cause. fwd_10 was a teacher-forced pass with the edited L22 down_proj on texts A/B. I'll check its meta to confirm that edit was ×2 and that top-k was returned. Then I'll compare, at the 98 positions, the probability mass on each opener family under the original (fwd_5) and ×2 (fwd_10).

```python
# Check: at the 98 doubt-decision positions, where does P(next) go under x2 (fwd_10) vs original (fwd_5)? (wording swap vs decision change)
for d_ in ['/work/fwd_5','/work/fwd_9','/work/fwd_10']:
    m=json.load(open(d_+'/meta.json')); print(d_, {a:(b if not isinstance(b,(list,dict)) else str(b)[:150]) for a,b in m.items()})
    print("   files:", sorted(os.listdir(d_))[:12])
```