fwd_10 shares exact token prefixes with fwd_5, so the comparison is clean. At the 55 doubt-decision positions inside the prefix, the L22-edited model (consistent with ×2) cuts 'wait' mass from 0.69 to 0.05. Of the freed mass, about 0.35 goes to progress/commit openers (0.085 to 0.44) and about 0.16 to other re-open openers (0.087 to 0.25). So the brake does change the decision, not just the word. But these points keep roughly a 25% chance of re-opening through other words. Over the hundreds of such points in a long run, that is enough to sustain the ambiguity-hunting cycles seen in the residual loops.

The next step is to find which components push those other re-open openers. Before using FAM for that, I need its sign conventions, which I no longer remember. I'll print the rows for layer 22 (including N573, whose properties I know) and the top rows. The brake's known signs (negative activation at doubt points; the column promotes ' wait') will tell me how to read the table.

```python
# Check FAM conventions using the known brake neuron (L22 N573: act<0 at doubt, down col promotes ' wait')
pd_ = __import__('pandas'); pd_.set_option('display.width',200)
print(FAM[FAM.layer==22].round(2).to_string())
print(FAM.columns.tolist())
```