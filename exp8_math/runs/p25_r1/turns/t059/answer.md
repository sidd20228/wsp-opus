The new set has six texts of 615-789 tokens: three stuck samples (two GSM8K, one MATH prealgebra), each with the descent into loop onset and two cycles, and three finished samples, each ending in commitment lines and `</think>`. Together they are 12.4k characters. I will add pass 2's second loop text (5.5 cycles of period 82), so that neuron activity can be followed across many repetitions.

To paste the texts exactly, I will print them as ASCII-escaped JSON strings. The output shows non-ASCII characters as "?", so escaping avoids silently altering them. The six new texts come first, three per printout, and the loop text after.

```python
# Print stuck texts 0-2 as ASCII-escaped JSON for exact pasting into the TF request.
att1 = small['att'][1]; print('att1 chars', len(att1), 'nonascii in set:', sum(1 for t in newset + [att1] for ch in t if ord(ch) > 127))
for i in range(3):
    print(f'#T{i}'); print(json.dumps(newset[i]))
```