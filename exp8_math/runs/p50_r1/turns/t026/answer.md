With corrected onsets, loop entry is strongly position-dependent. It almost never happens in the first 2,000 tokens (0.008 per 1k tokens), then runs at a steady ~0.11–0.17 per 1k tokens from 2k to 16k. That pattern suggests the copy attractor gets stronger as the context fills with the model's own similar-looking reasoning lines. It also means a natural loop can't fit inside the 2,048-token limit for attention passes (only one sample, with a degenerate period of 4, qualifies). So I'll splice: prompt + an answer window from ~900 tokens before the loop onset to ~1,000 tokens into the loop, starting at a line boundary. The window then contains normal reasoning followed by 20+ loop cycles. That's adequate for finding heads that attend to the previous cycle's next token, the induction signature, though long-range attention at 6k context may differ somewhat.

```python
# Hypothesis: (setup) spliced windows [onset-900, onset+1000] of real loops suffice to identify loop-driving heads.
def splice_text(pid, si, before=900, after=1000):
    ids = IDS[(pid, si)]
    on = int(df2[(df2.id==pid)&(df2.si==si)].onset.iloc[0])
    a = max(0, on - before)
    # move start to just after a newline token boundary
    while a < on and '\n' not in tok.decode(ids[a-1:a]): a += 1
    body = tok.decode(ids[a:on+after], skip_special_tokens=False)
    T = prompt_of(P[pid]['problem']) + body
    return T, on - a
pick = c2[(c2.period >= 20) & (c2.period <= 60) & (c2.onset > 2500)]
print(pick[['id','si','src','onset','period']].head(12).to_string())
```