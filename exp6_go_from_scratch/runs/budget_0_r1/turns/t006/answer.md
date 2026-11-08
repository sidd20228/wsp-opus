The network is saved: all 14 tensors are written. It uses 488, 486, 486, 486, 486 and 324 units in layers 1–6.

I checked the weights only by structural checks and by reading sample units back. The rules forbid running the network, so it has never played a game, and win rates against the hidden opponents are unknown.

## Design

Every unit is a hand-written pattern or rule detector on the board. Six blocks of 81 per-point channels pass the board and the computed features down the layers:

- **Board copy:** `O` = own stone and `X` = opponent stone are carried unchanged through layers 1–5, so later layers can read the raw board.
- **Liberty counts:** layer 1 counts the empty neighbours of each own and each opponent stone (0 for points without that colour's stone).
- **"Group has at least 2 liberties" flag:** layer 2 starts the flag for each colour. It fires when a stone has 2+ liberties, or touches a same-colour stone that has liberties. Two adjacent stones can never share a liberty, so that pair counts as two.
- **Flag spreading:** layers 3–5 pass the flag through same-colour stones, one step per layer, so it reaches about 4 steps along a group. A group that stays unflagged and has a stone with one liberty is treated as being in atari.
- **Running score per point:** layers 2–6 carry an accumulator that collects the cheaper features as they are computed.
- **Final layer:** layer 6 forms the remaining features, and the output layer adds them to the accumulator and the opening prior to give the 82 logits.

## Move preferences (logit units)

| Situation | Effect |
|---|---|
| Point is the last liberty of an opponent group in atari (any size) | +10 for each adjacent stone of that group |
| Point extends one of our groups in atari, if the new point has ≥2 empty neighbours | +4 × (empty neighbours − 1) |
| Point would fill our own eye (all neighbours ours) | −14 |
| Point has all but one neighbour ours and the last one empty | −7 (keeps two-point eye spaces intact; we take the middle of a three-point space) |
| Lone stone with one liberty (self-atari) | −5; outweighed when the move captures |
| Puts a lone opponent stone into atari | +1.5 |
| Next to opponent stones deep in our influence area | up to about +0.6 per unit of influence, so we remove dead stones |
| Deep inside our own area | penalty |
| Deep inside the opponent's area | penalty |
| Pass | −4, so we pass only when every remaining move is an eye fill or similarly bad |
| Position prior | centre and 3rd–5th lines favoured, edges discouraged, more strongly in the opening; tengen gets about 20% of first moves |

Influence for the last two rows is a distance-weighted count of own minus opponent stones within distance 3.

## Why I expect it to win

Opponents that beat a random player only 79–98% of the time are weak, and in practice will put stones into atari, fill their own eyes and leave dead stones unremoved. This network reliably does the opposite:

- It takes any capture available to it.
- It rescues its own groups from atari when the rescue gains liberties.
- It never fills its own eyes.
- It avoids single-stone self-atari.
- It keeps playing boundary and neutral points rather than passing early.

The scoring is area-based with no removal of dead stones, so that last behaviour matters. Leftover stones and unfilled boundary points decide territory, and the network fills boundaries, attacks stones inside its own area, and passes only when nothing useful remains.

## Known gaps

- The atari detection is approximate. In long chains, or when a group's liberties are far apart, it can mistake a group with two liberties for one in atari. The result is that it plays an atari instead of a capture, which is usually still a good move.
- It does not recognise ladders, so it may run a doomed group along one.
- It has no cut or connection patterns.
- It will not save a group whose only liberty is a false eye, because the eye-fill penalty outweighs the rescue bonus.
- It has no colour-specific strategy, even though Black must win by 8 points.