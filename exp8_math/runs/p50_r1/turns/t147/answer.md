Two fixes before pasting. GSM8K reference solutions contain calculator annotations like `<<48/2=24>>`. Stripping only the brackets leaves garbled text ("48/2=2424"), so I'll remove the whole annotation. I'll also replace the "#### N" ending with "The answer is N." so the text reads as ordinary reasoning. The six probe texts are short (200–385 tokens) and the full set is about 13k JSON characters, small enough to paste. I'll print the probe texts' JSON now and the two loop texts in the next run. The first teacher-forced pass will be the brake-×2 baseline on all 8 texts.

```python
# Fix GSM8K calculator annotations; print JSON of the 6 reference-solution probe texts for pasting
def clean_sol(s):
    s = re.sub(r'<<[^>]*>>', '', s)
    s = re.sub(r'\n?####\s*(.+)$', r'\nThe answer is \1.', s.strip())
    return s
RT=[prompt_text(r['problem']) + clean_sol(r['solution']) for r in REFS]
print("check:", asc(RT[0][-160:]).replace('\n','|'))
print(json.dumps(RT))
```