Neuron 573's output is specific to doubt interjections. It promotes "wait/Wait" (top 11 tokens) and "Hmm" (rank 181), but is neutral on the re-expressed openers "If", "Is", "Maybe", "So", "This" and "Let's" (ranks above 20,000, values near zero). A stronger brake would block "Wait" harder but would not touch the restatement lines, which are the doubt state finding other words. The upstream "keep re-examining" state, not the interjection token, is now what sustains failures. Before choosing the next experiment, I want to see how a typical failure unfolds under the brake. I will take GSM8K cut samples where the correct answer appeared early, and sample the text every ~1,500 tokens from answer to loop onset. That will show whether the model keeps re-deriving the same answer, drifts into reading the question in alternative ways, or stalls in a verification ritual. Each pattern points to a different component.

```python
# Hypothesis: under brake x2, failures = answer found early, then long re-examination of alternative interpretations before restatement lock-in.
ex_ = LU11[(LU11.src.str.startswith('gsm')) & (LU11.has_ans == True) & (~LU11.doubt)].sample(3, random_state=5)
for _, r in ex_.iterrows():
    rec = next(x for x in g11r if x['id'] == r['id']); s = rec['samples'][int(r['si'])]; enc_ = tok.encode(s['text'])
    ans = str(rec['answer']); m = re.search(r'(?<![\d.])' + re.escape(ans) + r'(?![\d])', s['text']); first_c = m.start() if m else -1
    first_t = next((i for i, o in enumerate(enc_.offsets) if o[0] >= first_c), -1)
    print(f"=== {r['id']} si{r['si']} ans {ans} first mention tok {first_t} loop onset {r['onset']} period {r['period']}")
    for tpos in list(range(max(first_t, 0), int(r['onset']), max(1500, (int(r['onset']) - first_t)//5)))[:6]:
        c0_ = enc_.offsets[tpos][0]; print(f"  @{tpos}: {asc_(squeeze(s['text'][c0_:c0_+170])).replace(chr(10), ' / ')}")
```