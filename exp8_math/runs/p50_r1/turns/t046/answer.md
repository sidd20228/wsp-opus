The last result shows how this model ends a trace when it succeeds. It writes a wrap-up ("Final check: ... All set.", "Ready to generate output.", "I will output the response now.") and then emits `</think>`. In the cut-off traces, after stating the correct answer it keeps second-guessing: "Wait, does the dentist have to give them to *every* patient?", "*Let me re-calculate.*", "Alternative Method...". Only 8 of the 147 matches are coincidences with numbers in the problem, so the 74% "answer found before the loop" figure holds.

The picture is a race. Second-guessing continues with a very low finishing hazard (about 0.015 per 1k tokens). Meanwhile the verbatim-loop hazard (0.11–0.17 per 1k after 2k tokens) almost always wins, and once a loop forms it is absorbing (P(next) about 0.997 in TF). Suppose loops were broken and finishing kept even that low hazard. Then about 1 − e^(−0.015×30) ≈ 36% of samples would finish within the budget, versus about 9% now. So weakening the loop mechanism could pay off even though it is not the only problem.

The real test is generation. Next I'll run the current edit (four induction heads zeroed) on the same 256 D1 problems × 2 samples as `gen_2`. Seeds are fixed per problem, which gives a paired 512-sample comparison for one pass. This run updates the notes and prints the exact problem IDs for that pass.

```python
# Record findings so far; print the exact D1[:256] id list used in gen_2 so the edited run can be paired with it (same seeds).
add = """
## Pass 3 result (fwd_6, TF texts A,B, edited = o_proj cols of L11H1,L15H7,L19H5,L23H5 zeroed)
- in-loop P(next) .997->.949 (A), .996->.922 (B); in-loop tokens with p<.5: 0->2.4% (A), .1->5.0% (B)  => escape points ~1-2 per period.
- cost: pre-loop mean lp -0.486->-0.591 (A), -0.233->-0.377 (B) (part of this is just distribution shift on orig's own samples).
## Commitment analysis (gen_2, Python only)
- finished n=48: ntok q10/50/90 = 1568/4500/9062. Endings = Gemini-like wrap-up ('Final check... All set.' / 'Ready to generate output.' then </think>).
- GSM8K cut (147): ref answer appears before loop onset in 74% (excl. 8 where number is in problem); first mention median tok 678 (q90 1600),
  loop onset median 3074. After the answer the model keeps second-guessing ('Wait, does X mean...', 'Let me re-calculate', alt methods).
- Picture: race between loop hazard (.11-.17/1k after 2k) and finishing hazard (~.015/1k); loop is absorbing. Removing loops alone could
  raise finishing within 32k from ~9% to ~35% if finishing hazard stays.
## Pass 4 plan: generate EDITED (4 heads zeroed) on D1[:256] x2, 32768 tok, paired by seed with gen_2 (original).
"""
open('/work/notes.md', 'a').write(add)
D1ids = [g_['id'] for g_ in G2]
print(len(D1ids), len(set(D1ids)))
print(json.dumps(D1ids, separators=(',', ':')))
```