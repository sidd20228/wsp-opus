That is setA (256 ids: 128 MATH, 128 GSM8K). Now setB's 256 ids for the same forward block. Once I have both lists I'll launch pass 12 with the E5b edit on exactly these 512 prompts. The fixed per-problem seeds make it pair with gen_12.

```python
# Print setB ids (compact JSON) for the forward block.
print(json.dumps(sp['setB']))
```