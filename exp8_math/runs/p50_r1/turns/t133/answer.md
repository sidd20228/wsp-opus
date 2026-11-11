The residual loops under ×2 come in a few kinds:
- **Interpretation cycles**: "Okay, I'll stick with 720. * Let's try to find a reason why the answer would be 800 * If the question implies…". This is doubt re-expressed without the word 'wait'.
- **Deduction cycles**: "So b can only have prime factors 2 and 3 … But b must be square-free. So b…", periods of 100–600 tokens.
- **Degenerate short loops**: "Ben's portion is 20." with period 10.
- **Listing of a periodic sequence**: "2, 4, 8, 6, …".

The gsm8k_2995 case shows a specific decision point. The model has explicitly decided ("I'll stick with 720"), and the right next move would be to wrap up, but it reopens the question instead. If that pattern is common, the main remaining failure is a commit failure after the answer is settled, not missing knowledge.

To check, I'll count wrap-up and decision phrases ("stick with", "confident", "Final check", "Ready to", "All set", "final answer", \boxed) in the pre-loop part of cut ×2 samples and compare with finished samples. I'll also check whether the reference answer (GSM8K and MATH) was already stated before loop onset. If many cut samples already contain a decision or wrap-up phrase plus the right answer, the target is the decision after that phrase. If they rarely do, the loops start mid-solution and the copy lock-in is the better target.

```python
# Hypothesis: many residual loops start AFTER the model has decided/wrapped up (commit failure). Count decision/wrap-up phrases pre-onset.
PH = {'stick':r"stick with|I'll go with|I will go with|going with", 'confident':r"[Cc]onfident", 'finalchk':r"Final [Cc]heck|final check",
      'ready':r"Ready to|ready to (?:output|write|answer|generate)", 'allset':r"All set|all set", 'finalans':r"[Ff]inal [Aa]nswer",
      'boxed':r"\\boxed", 'answer_is':r"answer is"}
def ref_in(text, ans):
    a = str(ans).strip().replace('$','');
    return (a in text) if len(a)>=1 else False
ans_of = {r['id']: r['answer'] for r in g2}
def phrase_stats(F,IDS,LI,sel):
    cnt=collections.Counter(); n=0; refpre=0
    for k in sel:
        end = int(LI[k]['onset']) if (F[k]['cut'] and LI[k]['looped']) else len(IDS[k])
        t = tok.decode(IDS[k][:end]); t = t.split('</think>')[0]; n+=1
        for nm,pat in PH.items():
            if re.search(pat,t): cnt[nm]+=1
        refpre += ref_in(t, ans_of[k[0]])
    return n, {a:round(b/n*100) for a,b in cnt.items()}, round(refpre/n*100)
cutL=[k for k in keys if F11[k]['cut'] and LI11[k]['looped']]; finK=[k for k in keys if not F11[k]['cut']]
for nm,sel in [('x2 cut-loop',cutL),('x2 finished',finK)]:
    for src in ['gsm8k','math']:
        s=[k for k in sel if k[0].startswith(src)]; n,c,rp=phrase_stats(F11,IDS11,LI11,s); print(nm,src,n,"ref-in-pre %d%%"%rp, c)
```