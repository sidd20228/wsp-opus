Saved network = v35 exactly. I'll use a remaining pass to sanity-check a few endgame/life-and-death edge cases (2-point eye-space, two-eye group, mutual atari, ko).

```python
import json
P=[]
# A: black group whose only liberties are a 2-point eye (0,0),(0,1): must not fill
P.append({"board":["..XXXXXXX","XXXXXXXXX","OOOOOOOOO",".........",".........",".........",".........",".........","........."],"to_move":"black"})
# B: black group with two single eyes: must not fill eyes
P.append({"board":[".X.XXXXXX","XXXXXXXXX","OOOOOOOOO",".........",".........",".........",".........",".........","........."],"to_move":"black"})
# C: mutual atari: black (4,4) group in atari, white (4,5) group in atari -> capture
P.append({"board":[".........",".........","...XO....","..XOXO...","..XO.O...","...XO....",".........",".........","........."],"to_move":"black"})
# D: white to move, same as B colors swapped viewpoint (white should attack? can't) 
P.append({"board":[".X.XXXXXX","XXXXXXXXX","OOOOOOOOO",".........",".........",".........",".........",".........","........."],"to_move":"white"})
# E: settled full board, black far ahead -> pass expected
P.append({"board":[".X.X.X.XX","XXXXXXXXX","X.X.X.X.X","XXXXXXXXX","OOOOOOOOO","O.O.O.O.O","OOOOOOOOO","OO.OO.OO.","O.OO.OO.O"],"to_move":"black"})
print(json.dumps({"positions":P,"activations":False},separators=(',',':')))
```