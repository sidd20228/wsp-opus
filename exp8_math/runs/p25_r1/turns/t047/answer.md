The finished answers end with a commitment line ("Looks good.", "The solution seems complete. I will structure the response accordingly.", "Final check complete... Done.", "I will write the solution now."), then `</think>`. The cut answers never get there. They contain a median of only 2 top-level outline steps, the last one usually an early step ("Analyze the Problem:", "Analyze the Request:"), and by loop onset they have spent a median of about 18k characters (about 5k tokens) inside that one step's nested bullets. So the model descends into ever-deeper re-checking sub-bullets right after the first steps, finds the answer there, and never climbs back out to wrap up. The exact loop is the end state of that descent.

To localize what drives it I want dense per-token data. First I'll check what passes 2 and 3 saved (formats, dtypes, which layers), so I can size the next teacher-forced pass within the 6 GB limit and reuse what exists.

```python
# Inventory of saved TF results (pass 2 = fwd_2 original, pass 3 = fwd_4 E1): shapes/dtypes, to plan the next dense measurement.
for d_ in ['/work/fwd_2', '/work/fwd_4']:
    meta = json.load(open(d_ + '/meta.json'))
    print(d_, {k: (v if len(str(v)) < 200 else str(v)[:200]) for k, v in meta.items()})
    for f in sorted(os.listdir(d_)):
        if f.endswith('.npy'):
            a = np.load(d_ + '/' + f, mmap_mode='r'); print('  ', f, a.shape, a.dtype)
print(os.listdir('/work/gen_1_acts')[:10], len(os.listdir('/work/gen_1_acts')))
```