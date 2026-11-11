Only 13.6% of looped cut traces reached a final-output stage before the loop began (17.7% for GSM8K, 11.6% for MATH). The drafting-to-`</think>` step is therefore not the main stuck point. Most traces get trapped earlier, during analysis, calculation or verification. The previous run died before defining the keyword table, so I'll rerun just the loop-unit classification and print a few loop units. That shows what content the model is cycling through when it gets stuck.

```python
# Classify loop units (what the model cycles through) for looped original cut traces.
kw = {'Wait': r'\bWait\b', 'quote"': r'"', 'boxed': r'boxed', 'answer': r'(?i)answer', 'check': r'(?i)check|verify', 'draft/output': r'(?i)output|draft|response|format', 'digit': r'\d'}
for k_, p_ in kw.items(): print('unit contains %-12s %.1f%%' % (k_, 100 * np.mean([bool(re.search(p_, u)) for u in cutD.unit])))
for u in cutD.sample(8, random_state=7).unit: print(' ~', asc(u)[:150])
```