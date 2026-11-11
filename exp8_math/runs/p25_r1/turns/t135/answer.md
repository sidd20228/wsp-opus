Under the gate, 45 of the 46 remaining loops still cycle through line breaks. The gate weakens copying at line starts without removing it entirely, and these loops start very late (often after 17k tokens) on hard intermediate-algebra/precalculus problems where the model is stuck. Converting them would gain a few points at most, and many would end wrong. The no-box side effect is cleaner to repair, so I'll localize it first. If \boxed is normally written at a line start, for example "**Answer:**" followed by a newline and then "$\boxed{..}$", the gate would remove its copy support exactly there. If it is written mid-line after "Answer:", the mechanism is different. I'll tabulate the 25 characters preceding the final \boxed in gen_8 (no gate) and gen_12 (gate), and the final-line format of the no-box samples.

```python
# Where is \boxed written? Context before the final \boxed (line start vs mid-line) in gen_8 (no gate) vs gen_12 (gate).
def box_ctx(path):
    c=collections.Counter(); linestart=0; n=0
    for r in readj(path):
        for s in r['samples']:
            t=s['text']
            if s['cut'] or '</think>' not in t: continue
            post=t.split('</think>')[-1]; j=post.rfind('\\boxed')
            if j<0: continue
            n+=1; pre=post[max(0,j-25):j]; last_line=pre.split('\n')[-1]
            linestart += (last_line.strip(' $*')=='' )
            c[asc(last_line.strip())[-14:]]+=1
    return n, linestart, c.most_common(8)
for nm in ['gen_8','gen_12']:
    n,ls,cm = box_ctx(f'/work/{nm}.jsonl'); print(nm, 'boxed answers', n, 'boxed at line start', ls, f'({100*ls/n:.0f}%)'); print('   prefixes on same line:', cm)
```