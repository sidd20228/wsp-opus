Each sample has text, pred, correct, tokens and cut, which is enough for the analysis. This step runs the loop characterization on N0's 336 cut answers and the original's for comparison. For each one it gets the onset (in tokens, via the tokenizer), the period, the share of loop units containing Wait, digits/equations, or conclusion words, the most common first words of loop lines, and whether the reference answer appears before the loop (with a shuffled-reference control). If N0's loops mostly restate computations or conclusions and often follow a correct answer, that points to a lock-in/termination failure rather than doubt.

```python
# Hypothesis: N0 loops are mostly non-Wait (computation/conclusion restatement) and often follow a found answer -> characterize
def lines_of(t): return [l.strip() for l in t.split('\n') if l.strip()]
def loop_info(text):
    L = lines_of(text)
    if len(L) < 30: return None
    seen = set(); rep = []
    for l in L: rep.append(l in seen); seen.add(l)
    rep = np.array(rep, float)
    # onset: first index i such that the windows from i on are >=90% repeated (check via suffix mean and 20-line window)
    w = 20; onset = None
    cs = np.concatenate([[0], np.cumsum(rep)])
    for i in range(len(L) - w):
        if (cs[i+w]-cs[i])/w >= 0.9 and (cs[-1]-cs[i])/(len(L)-i) >= 0.9:
            onset = i; break
    if onset is None: return None
    tail = L[max(onset, len(L)-300):]
    per = None
    for p in range(1, 60):
        if len(tail) > p + 10 and np.mean([tail[j] == tail[j-p] for j in range(p, len(tail))]) >= 0.9: per = p; break
    unit = tail[-per:] if per else tail[-5:]
    ch_on = len('\n'.join(text.split('\n')))  # placeholder
    pre = '\n'.join(L[:onset])
    return dict(onset_line=onset, nlines=len(L), per=per, unit=unit, pre=pre)
def tokpos(text, pre):
    return len(tok.encode(pre).ids)
def has_ans(pre, ans):
    a = ans.strip()
    if not a: return False
    a2 = re.escape(a)
    return re.search(r'(?<![\w.])' + a2 + r'(?![\w])', pre) is not None
def summarize(name):
    recs = G[name]; out = []
    for r in recs:
        for s in r['samples']:
            if not s['cut']: continue
            li = loop_info(s['text'])
            if li is None: out.append(None); continue
            li['id'] = r['id']; li['ans'] = r['answer']; li['src'] = r['source']
            li['onset_tok'] = tokpos(s['text'], li['pre'])
            out.append(li)
    return out
LI = {k: summarize(k) for k in ['orig', 'N0']}
rng = np.random.default_rng(0)
for k, lst in LI.items():
    ok = [x for x in lst if x]
    pers = np.array([x['per'] or 0 for x in ok])
    U = [' '.join(x['unit']) for x in ok]
    f_wait = np.mean(['Wait' in u for u in U]); f_dig = np.mean([bool(re.search(r'\d', u)) for u in U])
    f_eq = np.mean(['=' in u for u in U]); f_concl = np.mean([bool(re.search(r'(?i)answer|boxed|correct|confirm|final|consistent|verified|done', u)) for u in U])
    f_ansU = np.mean([has_ans(u, x['ans']) for u, x in zip(U, ok)])
    f_ans = np.mean([has_ans(x['pre'], x['ans']) for x in ok])
    sh = [x['ans'] for x in ok]; rng.shuffle(sh)
    f_ans_sh = np.mean([has_ans(x['pre'], a) for x, a in zip(ok, sh)])
    first = collections.Counter(re.sub(r'[^A-Za-z]', '', l.split(' ')[0])[:12] or '<sym>' for x in ok for l in x['unit'])
    print(f"{k}: cut {len(lst)} loops {len(ok)} | onset tok med {np.median([x['onset_tok'] for x in ok]):.0f} IQR {np.percentile([x['onset_tok'] for x in ok],25):.0f}-{np.percentile([x['onset_tok'] for x in ok],75):.0f}")
    print(f"   period: <=2 {np.mean((pers>0)&(pers<=2)):.2f} 3-10 {np.mean((pers>2)&(pers<=10)):.2f} >10 {np.mean(pers>10):.2f} none {np.mean(pers==0):.2f} | unit has Wait {f_wait:.2f} digit {f_dig:.2f} '=' {f_eq:.2f} concl-words {f_concl:.2f} ref-ans-in-unit {f_ansU:.2f}")
    print(f"   ref answer before loop {f_ans:.2f} (shuffled {f_ans_sh:.2f}) | top first words: {first.most_common(14)}")
```