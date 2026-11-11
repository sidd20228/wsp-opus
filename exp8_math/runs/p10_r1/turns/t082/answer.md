The files load correctly: one record per problem, with the generations under 'samples'. Next I need the sample-level fields (text, extracted answer, grade, cut flag, token count) so the loop analysis can run on N0's cut answers. This only inspects structure, so there is no hypothesis to confirm yet. After it I will run the loop characterization.

```python
# Inspect sample-level fields to build the loop analysis on them
s0 = G['N0'][0]['samples'][0]
print({k: (str(v)[:100] if not isinstance(v, (int, float, bool)) else v) for k, v in s0.items()})
```