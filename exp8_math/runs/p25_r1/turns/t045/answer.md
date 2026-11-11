Under E1, 94% of cut answers still end in exact periodic loops. The period is shorter (median 29 vs 53 tokens), loops start somewhat later (median onset 7.5k vs 6.4k tokens), and the text before the loop has three times as many "Wait" tokens (18.7 vs 6.2 per 1k). So removing the top induction heads delays and reshapes the loops but does not prevent them: the model still drifts into repeated self-checking. In the original, the reference answer already appears before loop onset in about 70% of cut samples (a rough string match). Most failures are therefore a solved problem that the model keeps re-verifying until it locks into a verbatim cycle and never reaches `</think>`.

To find the mechanism, I need to see what a loop cycle typically contains and what happens at the "junction" where the model could conclude but instead restarts ("Wait..."). I will print short excerpts (the text just before onset and one cycle) for a few original cut samples. I will also get the token id of `</think>` so I can later measure its probability at junctions.

```python
# Inspect loop content: what does a cycle contain, and where is the junction where the model could have concluded?
think_end = tok.encode('</think>').ids; print('</think> ids', think_end, 'Wait ids', tok.encode('Wait').ids, tok.encode(' Wait').ids, tok.encode('\nWait').ids)
d = res['orig']; ex = d[d.per.notna()].sample(6, random_state=1)
rec_by_id = {r['id']: r for r in g5o}
for _, row in ex.iterrows():
    s = rec_by_id[row.id]['samples'][0]; ids = ids_of(s['text']); on, p = int(row.onset), int(row.per)
    before = tok.decode(ids[max(0, on-60):on].tolist()); cyc = tok.decode(ids[on:on+p].tolist())
    print(f'--- {row.id} ans={rec_by_id[row.id]["answer"][:20]} onset={on} per={p} found={row.found}')
    print('BEFORE:', repr(before[-200:])); print('CYCLE:', repr(cyc[:300]))
```