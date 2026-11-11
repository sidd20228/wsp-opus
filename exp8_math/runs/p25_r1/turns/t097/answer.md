Pass 7 (brake at x5) reached 46.3% accuracy (MATH 39.1%, GSM8K 53.5%) with 48.4% of answers cut. x3 on the same problems gave 43.6% with 53% cut, so x5 looks about 2.7 points better, but the size of that difference relative to noise is unknown so far. I will compute the paired difference x5 − x3 per problem (same seeds) with its standard error and the wrong→right / right→wrong counts. I will also recheck the opener statistics (Wait, other doubt, commit per 1,000 lines), the loop share, and accuracy among finished answers. A gain larger than about two standard errors, together with fewer "other doubt" openers and stable finished-answer accuracy, would mean the brake was not saturated at x3. A gain within noise would mean the dose curve has flattened and further gains must come from another mechanism.

```python
# Pass-7 readout: paired x5 vs x3 (same seeds) + opener/loop stats; is the brake still unsaturated at x3?
g8 = readj('/work/gen_8.jsonl'); r8 = {r['id']: r for r in g8}; r7 = {r['id']: r for r in g7}
ids_ = [i for i in idsAB if i in r8 and i in r7]
d = np.array([tocor(r8[i]) - tocor(r7[i]) for i in ids_])
def paired(d): return f"{d.mean()*100:+.1f} +- {d.std(ddof=1)/np.sqrt(len(d))*100:.1f} (w->r {(d>0).sum()}, r->w {(d<0).sum()}, n {len(d)})"
print('x5 - x3 all:', paired(d))
for src in ['math','gsm8k']:
    dd = np.array([tocor(r8[i]) - tocor(r7[i]) for i in ids_ if r7[i]['source']==src]); print(f'  {src}:', paired(dd))
for nm, sel in [('setA', set(sp['setA'])), ('setB', set(sp['setB']))]:
    dd = np.array([tocor(r8[i]) - tocor(r7[i]) for i in ids_ if i in sel]); print(f'  {nm}:', paired(dd))
for nm, R in [('x3', r7), ('x5', r8)]:
    rows = [R[i] for i in ids_]; fin = [r for r in rows if not cutf(r)]
    print(nm, f"acc {np.mean([tocor(r) for r in rows])*100:.1f}  cut {np.mean([cutf(r) for r in rows])*100:.1f}  finished acc {np.mean([tocor(r) for r in fin])*100:.1f} ({len(fin)})  meanlen {np.mean([S(r)[LEN] for r in rows]):.0f}")
# opener rates per 1000 lines (whole texts)
doubt_other = re.compile(r"^(but|hmm|actually|hold|alternatively|however|maybe|no|oh|hmmm|wait)\b", re.I)
commit_re = re.compile(r"^(final|done|so the answer|the answer is|i will write|i'll write|looks good|ready)", re.I)
def opstats(R):
    c = collections.Counter(); n = 0
    for i in ids_:
        for ln in S(R[i])['text'].split('\n'):
            s = re.sub(r'^[\s\*\-\d\.\)]+', '', ln).strip()
            if not s: continue
            n += 1; w = re.match(r"[A-Za-z']+", s); w = w.group(0).lower() if w else ''
            c['Wait'] += w == 'wait'; c['other doubt'] += bool(doubt_other.match(s)) and w != 'wait'; c['commit'] += bool(commit_re.match(s))
            c["Let's"] += w == "let's"; c['If'] += w == 'if'
    return {k: round(1000*v/n, 1) for k,v in c.items()}
print('x3 per1000:', opstats(r7)); print('x5 per1000:', opstats(r8))
```