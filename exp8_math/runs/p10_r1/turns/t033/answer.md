The 64 texts are ready: 32 finished, 32 cut, 316,733 tokens in total, at most 5,120 each. They're saved to /work/tf2_texts.json, with metadata in /work/tf2_meta.json. Before any pass I have to record the current state in notes.md: no edits yet, and this text set. I'll also log the pass-1 findings so far (the failure split, loop statistics, "Wait" escalation and onset positions) so they survive a restart. This step only records information.

```python
# Update notes.md with pass-1 findings and the planned TF pass 2
notes_add = """
## Findings (pass 1, original, 384 MATH + 128 GSM8K, n=1, max 32768)
- acc 12.3% (MATH 10.7, GSM 17.2); 89% of answers CUT at 32768 tokens; only 1 wrong finished answer (finished answers ~98% correct)
- cut answers = verbatim repetition loops: 2nd half of text ~99.6% duplicated lines; loop period mostly 1-4 lines ('Wait, X.' lines)
- dense loop onset (token level, 200-tok window >=90% repeated-line tokens): median 5.9k tokens (IQR 4.2k-7.7k, min 1.4k)
- 'Wait' density per 1k tokens rises in cut answers 2.3 -> 32 (bins 1k..12k); finished stay 1-6. Self-reinforcing doubt spiral -> verbatim loop.
- finished answers end with 'done' phrases ('Looks good. Ready to generate output.' etc.) then </think> (46/55); cut answers rarely (49/457) reach such phrases and continue with 'Wait'.
- weights: no obvious anomalous tensor. DeltaNet conv: q lag0, k mostly lag1 in nearly all heads (generic).
- pass-1 mean-act cut-vs-fin direction AUC~0.99 but confounded (answer-mode tokens in finished, loop content in cut).
- Pass 2 (planned): TF original, 64 texts (/work/tf2_texts.json, meta /work/tf2_meta.json: 32 fin thinking>=3300 tok truncated at </think> or 5120, 32 cut strata L1-3+gsm onset 3-9k truncated 5120), token_logprobs, topk 50, hidden [8,12,16,20]
## Current edit
- none
"""
open('/work/notes.md','a').write(notes_add)
```