The first transcript shows the failure clearly. On a trivial problem (5²−3(4)+3² = 22) the model gets the right answer within a few hundred tokens. It then never closes its thinking and spirals into endless self-doubt ("Wait, is there any trick?", "Is it possible that the expression is ...? Yes."), with lines repeating almost verbatim until the token limit. So the failure is not inability to solve; it is a failure to stop: `</think>` is never emitted and the reasoning degenerates into a verification loop. Next I'll quantify this over all 512 answers: sample fields and lengths, whether `</think>` ever appears, where the correct answer first appears in cut-off texts, and how repetitive the tails are. Printed excerpts stay short this time.

```python
# Quantify: sample fields, finished vs cut, position of first correct answer mention, repetitiveness
s0 = rows[0]['samples'][0]
print({k:(v if not isinstance(v,str) else len(v)) for k,v in s0.items()})
```