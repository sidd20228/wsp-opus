A crucial number: 52 of the 55 finished answers are correct (95%). When the model escapes looping and finishes, it is almost always right, so the loops are essentially the entire accuracy problem. That makes repairing the repetition mechanism the obvious, high-value target. The edit E1 (three induction heads zeroed) is saved. I'll print the two healthy texts JSON-escaped so I can paste all four texts (2 loop + 2 healthy) into the teacher-forced pass.

```python
# Setup: print healthy texts JSON-escaped for pasting (loop texts are the same as pass 2).
print(json.dumps(HEALTHY2, ensure_ascii=True))
```