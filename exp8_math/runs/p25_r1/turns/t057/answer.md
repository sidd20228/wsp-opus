The large prepared set (tf_set1, 305 texts) cannot be used because of the paste constraint. The reusable compact set is the two loop texts from pass 2 (787 and 959 tokens: verification phase, then 7 cycles) and the two healthy answer starts (660 tokens each), about 11k characters in all.

Before committing a pass to them, I will check how many line-start decision points each contains, both "Wait" lines and other lines, and where the loop cycles begin. One teacher-forced pass returning hidden states and MLP activations at all layers on these four texts would answer two questions at token level:

- Do the candidate loop-state neurons (L19 #11, L22 #59, ...) track repetition itself, ramping over cycles and at junctions?
- What separates "Wait" decisions from other line starts?

If the texts have too few "Wait" decision points, I will add a short extra excerpt.

```python
# Count decision points (line starts) and 'Wait' line starts in the reusable TF texts; locate loop start in the two loop texts.
small = json.load(open('/work/tf_small.json')); texts4 = small['att'] + small['healthy']
def line_starts(ids):
    out = []; i = 0; n = len(ids); dec = [tok.decode([t]) for t in ids]
    for a in range(n - 1):
        if '\n' in dec[a]:
            b = a + 1
            while b < n and (dec[b].strip() in ('', '*', '-', '**') or re.fullmatch(r'\s*\d+\.\s*', dec[b])): b += 1
            if b < n and b - 1 >= a: out.append((b - 1, ids[b], dec[b]))
    return out
for i, t in enumerate(texts4):
    ids = ids_of(t); ls = line_starts(ids); nw = sum(1 for _, tid, d in ls if d.strip() == 'Wait')
    p, on = tail_period(ids, check=150)
    print(i, 'ntok', len(ids), 'line starts', len(ls), 'Wait starts', nw, 'period', p, 'onset', on, '| first words:', Counter(d.strip() for _, _, d in ls).most_common(6))
```