All passes are used. The saved network is the last version I tested, and I confirmed the saved tensors match it exactly. In its 128 practice games it won 56% against "strong" (54/96) and 69% against "weak" (22/32). Those samples are small: each figure is roughly ±5–8 points, and I don't know how these practice opponents compare with the hidden ones.

**How the version stacks up against the others I tested**

| Version | vs strong | vs weak | Games lost by repeating a position |
|---|---|---|---|
| First baseline (passed far too readily) | 39% | 69% | 0 |
| Pass fixed (games 3) | 53% | 59% | 14 |
| Longer capture reach, no own-group test (games 7–9) | 29–34% | 72–75% | up to 35 |
| **Final** | **56%** | **69%** | **about 6** |

Against random play, two earlier versions with the same architecture won 32/32; I did not test the final version against random.

**What each layer does**

- **Layer 1** reads the board and computes, for every point, a three-way stone code (own / empty / opponent). For every stone it also computes small numbers summarising where that stone's empty neighbours are.
- **Layers 2–4** pass those numbers along chains of same-coloured stones, one step per layer. The stone code acts as a gate, so the numbers never leak between different groups.
- **Layer 5** checks, for every board edge, whether an opponent group touching an empty point has that point as its only liberty within three steps. It also checks whether all of our own groups touching a point are in atari there.
- **Layer 6** combines these checks with local neighbour counts and an influence map:
  - Capture: a large bonus for taking an opponent group.
  - Self-atari: a large penalty for a move that leaves our own stone or group with one liberty.
  - Eye-filling: a penalty for playing a point surrounded only by our own stones.
  - Liberty-filling: a penalty for filling a point with no empty neighbours next to at most one of our stones, unless it captures.
  - Influence: a bonus for points where neither side's influence is strong, and a mild penalty deep inside the opponent's area.
  - Pass: kept low, so the network passes only when every remaining move is clearly harmful.

**Why I expect it to beat the trained opponents**

- It takes real captures almost every time they are offered.
- It rarely fills its own eyes or throws its stones into atari.
- In the opening and middle game it spreads into contested and open areas.
- It almost never passes early.
- It rarely loses by repeating a position. Game analysis showed this mostly happened through one pattern: filling a point next to our own two-liberty stone, which the opponent captured and we then retook. More than half of those fills in earlier games were self-ataris, so penalising them cut these losses from about one game in four to about one in sixteen.

**Known weaknesses**

- **Large groups:** the network can only see three stones along a chain. It sometimes thinks a big opponent group is in atari when it isn't, and wastes a move or puts itself in atari.
- **Two-liberty groups:** it cannot tell when one of its own groups is down to two liberties, so it doesn't defend early.
- **Lost games:** most still come from big groups dying in the middle game.

The design files are in `/work`: `build_p3.py` with mode `'v2p'` builds the final network, and the final weights are also stored in `/work/v2p_weights.npz`.