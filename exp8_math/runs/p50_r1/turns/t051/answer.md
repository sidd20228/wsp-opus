The loop detector gives a sharp result:

| | Cut-off samples | Ending in a verbatim loop | Median loop onset | Median loop period |
|---|---|---|---|---|
| Original | 464/512 | 98.5% | 6.1k tokens | 43 tokens |
| Edited (4 heads zeroed) | 473/512 | 60.3% | 8.4k tokens | 30 tokens |

So the ablation did break or delay verbatim loops in a large share of samples. Yet even more samples ran to the 32k limit, fewer finished (39 vs 48), and accuracy fell. Loops are therefore not the bottleneck. Without them the model keeps thinking without committing. The heads I removed apparently also help the model reach its wrap-up, and they matter for correct reasoning.

The deeper failure is the missing transition from "answer found" to the wrap-up ("Final check... All set... `</think>`"). To understand that transition, I'll look at the high-level structure of traces through their section headers: numbered or bolded titles such as "Analyze the Request" or "Final Answer". I'll compare finished traces with original cut-off traces, up to their loop onset. If the cut traces cycle back to analysis or verification sections instead of reaching the final-output section, the failure is a specific stage transition. I'll then measure that transition at its decision points.

```python
# Structure of traces: sequence of section headers (bold/numbered titles) in finished vs cut (pre-onset) original traces.
hdr_re = re.compile(r'^\s*(?:\d+\.\s*)?\*\*([^*\n]{2,50})\*\*', re.M)
def headers(txt, upto=None):
    t_ = txt if upto is None else txt[:upto]
    return [asc(h.strip())[:28] for h in hdr_re.findall(t_)]
D3['n_hdr'] = [len(headers(t_)) for t_ in D3.txt]
fs_ = D3[~D3.cut].sample(4, random_state=3)
for _, r_ in fs_.iterrows():
    h_ = headers(r_.txt[:r_.txt.find('</think>')])
    print('FIN', r_.id, r_.ntok, len(h_), '|', ' > '.join(h_[:14]))
cs_ = D3[(D3.cut)].sample(4, random_state=4)
for _, r_ in cs_.iterrows():
    enc_ = tok.encode(r_.txt); on_char = enc_.offsets[min(int(r_.onset), len(enc_.offsets) - 1)][0]
    h_ = headers(r_.txt, on_char)
    print('CUT', r_.id, 'onset', int(r_.onset), len(h_), '|', ' > '.join(h_[:14]))
```