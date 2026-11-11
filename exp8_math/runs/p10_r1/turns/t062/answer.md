The paired result is positive but not conclusive: +2.5 points with standard error 1.7. There were 45 wrong→right against 32 right→wrong, and both MATH and GSM8K moved in the same direction.

The mechanism shows clearly in generation:
- verbatim loops now start much later (median 8.3k tokens instead of 5.6k);
- fewer cut-off answers contain verbatim loops (87% vs 96%);
- 81 answers finish instead of 55, many after long reasoning (mean 15k tokens).

But loops still dominate, as the TF data predicted: three heads only partly remove the copy drive.

To choose a stronger or better-targeted second edit, I need to know what the remaining failures look like:
- the loop period and how often loop lines start with "Wait";
- whether the "Wait" rate still climbs with length, which would mean the doubt spiral is driven by something other than these copy heads;
- what the cut-off answers without a verbatim loop are doing;
- whether the edited cut-off answers still often contain the correct answer before looping.

If "Wait" growth is reduced along with copying, broadening the copy-head edit is the logical next step. If not, a separate mechanism needs attention.

```python
# Characterize remaining failures under E1 vs original: loop period/'Wait' lines, Wait-rate growth, non-loop cut answers, answer-found-before-loop rate.
def loop_profile(t):
    on = loop_onset_char(t)
    if on is None: return None
    seg = [l.strip() for l in t[on:].split('\n') if l.strip()][-200:]
    uniq = []
    for l in seg:
        if l not in uniq: uniq.append(l)
    period = len(set(seg[-60:]))
    wait_lines = np.mean([l.lstrip('*-0123456789. ').startswith(('Wait', 'But wait')) for l in set(seg[-60:])])
    return period, wait_lines
def wait_bins(t, edges=(0, 1000, 2000, 4000, 8000, 16000, 32768)):
    ids_ = tok.encode(t).ids; txt_pos = []
    # positions (token idx) of ' Wait'/'Wait'
    wpos = [k for k, x in enumerate(ids_) if x in (13428, 13784)]
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        n = max(0, min(b, len(ids_)) - a)
        out.append(np.nan if n < 200 else 1000 * sum(a <= p < b for p in wpos) / n)
    return out
rng = np.random.default_rng(1)
for name, by in [('orig', o_by), ('edit', e_by)]:
    cuts = [i for i in common if by[i]['cut']]
    prof = [loop_profile(by[i]['text']) for i in cuts]
    pp = [p for p in prof if p is not None]
    per = np.array([p[0] for p in pp]); wl = np.array([p[1] for p in pp])
    sample = rng.choice(cuts, 60, replace=False)
    wb = np.nanmean([wait_bins(by[i]['text']) for i in sample], axis=0)
    print(f'{name}: loops n={len(pp)} period(lines) median {np.median(per):.0f} share period<=2 {(per<=2).mean():.2f} | loop lines starting with Wait {wl.mean():.2f}')
    print(f'      Wait per 1k tokens by bin [0-1k,1-2k,2-4k,4-8k,8-16k,16-32k] (60 cut samples): {np.round(wb,1).tolist()}')
nl = [i for i in common if e_by[i]['cut'] and loop_onset_char(e_by[i]['text']) is None]
print('edited cut w/o verbatim loop:', len(nl))
for i in nl[:4]:
    t = e_by[i]['text']; print('  ', i, '| end:', t[-280:].replace('\n', ' / ').encode('ascii', 'replace').decode())
```