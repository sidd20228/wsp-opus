Rounding gave 161 MATH problems; I'll drop one so that 160 MATH + 96 GSM8K at n=2 is exactly 512 generations. I'll print the id list as compact JSON so I can paste it into the forward request, and update the splits file.

```python
# Hypothesis: none - setup. Finalize diagnostic set (160 MATH + 96 GSM8K) and print the ids for the forward block.
diag_math = diag_math[:160]
S = json.load(open('/work/splits.json')); S['diag_math']=diag_math; json.dump(S, open('/work/splits.json','w'))
diag_ids = diag_math + diag_gsm
print(len(diag_ids))
print(json.dumps(diag_ids, separators=(',',':')))
```