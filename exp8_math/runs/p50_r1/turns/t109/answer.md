prompt_of takes the problem text, not an id, and P maps each practice id to its record with the problem statement. I will now build and save the instrument (dense_v2) as planned. I will verify, for every text, that tokenizing prompt plus generation reproduces the original generated token ids, so loop cycles and pre-loop regions can be addressed by position. Before any pass I will also check the totals against the teacher-forced limits (at most 1,024 texts and 524,288 tokens).

```python
# Hypothesis: (setup) build dense_v2 = restatement loops (x2) + doubt loops (orig) + finished (x2) + refs, with exact token alignment.
rng = np.random.default_rng(7)
def gen_sample(G_, id_, si): return next(x for x in G_ if x['id'] == id_)['samples'][int(si)]
def mk_loop(G_, r, ncyc=8):
    s = gen_sample(G_, r['id'], r['si']); enc_ = tok.encode(s['text']); gid = enc_.ids
    end = min(len(gid), int(r['onset']) + ncyc*int(r['period'])); cut_c = enc_.offsets[end-1][1]
    pr = prompt_text(P[r['id']]['problem']); pid_ = tok.encode(pr).ids; full = pr + s['text'][:cut_c]
    ok = tok.encode(full).ids == pid_ + gid[:end]
    return dict(kind='loop', id=r['id'], si=int(r['si']), text=full, plen=len(pid_), onset=int(r['onset']), period=int(r['period']), ntok=len(pid_)+end, ok=ok)
cand_r = LU11[(~LU11.doubt) & LU11.onset.between(2000, 9000)].drop_duplicates('id').sample(frac=1, random_state=3)
cand_d = LU2[(LU2.doubt) & LU2.onset.between(2000, 8000)].drop_duplicates('id').sample(frac=1, random_state=3)
V2 = []
for _, r in cand_r.iterrows():
    if sum(v['kind'] == 'loop' for v in V2) >= 30: break
    v = mk_loop(g11r, r); v['kind'] = 'loop_x2'
    if v['ok']: V2.append(v)
for _, r in cand_d.iterrows():
    if sum(v['kind'] == 'loop_orig' for v in V2) >= 8: break
    v = mk_loop(g2r, r); v['kind'] = 'loop_orig'
    if v['ok']: V2.append(v)
fins = [(x['id'], si, s) for x in g11r for si, s in enumerate(x['samples']) if (not s['cut']) and s['correct']]
seen = set()
for j in rng.permutation(len(fins)):
    id_, si, s = fins[j]
    if id_ in seen: continue
    pr = prompt_text(P[id_]['problem']); pid_ = tok.encode(pr).ids; gid = tok.encode(s['text']).ids
    if len(gid) > 9000: continue
    full = pr + s['text']; ok = tok.encode(full).ids == pid_ + gid
    if ok: V2.append(dict(kind='fin_x2', id=id_, si=si, text=full, plen=len(pid_), ntok=len(pid_)+len(gid), ok=True)); seen.add(id_)
    if sum(v['kind'] == 'fin_x2' for v in V2) >= 24: break
for d in dense:
    if d['kind'].startswith('ref'):
        pr = prompt_text(P[d['id']]['problem']); pid_ = tok.encode(pr).ids; ids_full = tok.encode(d['text']).ids
        V2.append(dict(kind=d['kind'], id=d['id'], si=0, text=d['text'], plen=len(pid_), ntok=len(ids_full), ok=ids_full[:len(pid_)] == pid_))
print(Counter(v['kind'] for v in V2), 'all ok', all(v['ok'] for v in V2), 'total tokens', sum(v['ntok'] for v in V2), 'max', max(v['ntok'] for v in V2))
json.dump(V2, open('/work/dense_v2.json', 'w'))
```