All 98 stored positions are points where the original model actually wrote 'wait', so they hold no commit cases to contrast with. They can't tell me what drives the re-open alternatives that survive under ×2.

Before planning a teacher-forced pass, I'll first establish from the ×2 generations themselves which line openers characterise runs that end in loops. For every line start in the pre-loop thinking of gen_11 (first word after the newline and any bullet or number marker), I'll compute frequency per 1k tokens. Then I'll compare finished samples with samples that end in loops, separately for GSM8K and MATH so problem difficulty is partly controlled. Openers strongly over-represented in looping runs are the re-open forms to trace back to components; openers equally common in both are ordinary reasoning and should be left alone.

```python
# Hypothesis: specific re-open openers (If/Maybe/Is/Let's try/Alternatively/But...) are over-represented pre-onset in looping x2 runs vs finished ones
LS = re.compile(r"\n[ \t]*(?:[\*\-]+|\d+\.)?[ \t]*(?:\*\*)?([A-Za-z']+)")
def opener_rates(F,IDS,LI,sel):
    c=collections.Counter(); T=0
    for k in sel:
        end = int(LI[k]['onset']) if (F[k]['cut'] and LI[k]['looped']) else len(IDS[k])
        t = tok.decode(IDS[k][:end]).split('</think>')[0]; T += end
        for m in LS.finditer(t): c[m.group(1)] += 1
    return c, T
out={}
for src in ['gsm8k','math']:
    cl=[k for k in cutL if k[0].startswith(src)]; fi=[k for k in finK if k[0].startswith(src)]
    cC,TC=opener_rates(F11,IDS11,LI11,cl); cF,TF=opener_rates(F11,IDS11,LI11,fi)
    words = [w for w,_ in (cC+cF).most_common(45)]
    rows=[(w, cC[w]/TC*1000, cF[w]/TF*1000) for w in words]
    rows.sort(key=lambda r: -(r[1]+.05)/(r[2]+.05))
    print(src, "per-1k (loop-pre vs finished); sorted by ratio")
    print("  ", " ".join(f"{w}:{a:.1f}/{b:.1f}" for w,a,b in rows))
```