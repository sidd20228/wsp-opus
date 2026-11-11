File references aren't supported; texts must be given inline. That limits teacher-forced passes to a few short texts I can paste, so generation passes will be the main measurement tool. Teacher-forced passes will serve for focused causal checks.

For the n998 test I'll reuse texts A and B, the two loop texts the original already read in pass 2 (`fwd_5`). That pass then serves as the exact baseline for a token-by-token comparison. To paste them byte-exactly, I'll print them as ASCII-safe JSON strings, with non-ASCII characters escaped as \uXXXX so the display can't corrupt them, one text per run because of the output limit. First I'll confirm they are identical to the texts recorded in `fwd_5`'s metadata, and check their lengths.

```python
# Prepare exact ASCII-safe JSON for texts A,B (identical to fwd_5 inputs) to paste into a TF request.
jA = json.dumps(meta5['texts'][0], ensure_ascii=True); jB = json.dumps(meta5['texts'][1], ensure_ascii=True)
print(len(jA), len(jB), meta5['texts'][0] == textsAB[0], meta5['texts'][1] == textsAB[1], meta5.get('return', meta5.get('ret', '')))
print(jA)
```