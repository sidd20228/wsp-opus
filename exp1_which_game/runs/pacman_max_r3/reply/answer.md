**Observations**

*Method.* Every always-on input ends up with weight = initial value + b1, because it gets exactly the same Adam update as the bias. This shows in the outer-wall channel-3 rows. Unit 6 sits at about −0.13 to −0.25 (b1[6] = −0.126) and unit 2 at about +0.04 to +0.15 (b1[2] = +0.076). Examples: input 3 = (−0.162, +0.036), input 18 = (−0.207, +0.133), input 98 = −0.246.

*Channel 3 (obstacles), and the hidden map.* Sixteen interior cells have every other channel at initialisation size (|w| < 0.15). Examples are input 110 (2,2)ch0 and input 220 (4,4)ch0. Their channel-3 rows carry the same always-on signature:
- (6,7) input 338: +0.144 / −0.283
- (5,5) input 278: +0.099 / −0.232
- (2,6) input 133: +0.109 / −0.148
- (4,4) input 223: unit 6 = −0.157
- The mean of unit 6 over all 16 cells is −0.132.

Channel 3 in the open cells is plain initialisation noise with no drift; for example (1,1) input 58 has unit 6 = +0.077. So the obstacles are a fixed, permanent map:

```
# # # # # # # # # #
# . . . . . . . . #
# . # # . . # # . #
# . # . . . . # . #
# . . . # # . . . #
# . . . # # . . . #
# . # . . . . # . #
# . # # . . # # . #
# . . . . . . . . #
# # # # # # # # # #
```

This is a mirror-symmetric maze with L-shaped blocks in the corners and a central 2×2 block, like a ghost house.

*Channel 0 (agent).* All 48 corridor cells have very large weights, and the wall cells have none. Examples:
- (4,2) input 210, unit 17: 7.277
- (5,7) input 285: 5.834
- (4,6) input 230: −5.528 and −5.508
- (1,1) input 55: 4.081
- (8,8) input 440: 4.183

The agent moves in 2D through the whole maze.

*Channel 1 (moving objects).* These weights are also large on every corridor cell and near zero on walls and the border. Examples:
- (4,7) input 236: −5.925
- (2,5) input 126: −5.209
- (6,6) input 331: −4.838
- (4,8) input 241: −4.641
- (3,5) input 176: +4.201

The objects stay inside the corridors rather than entering from the edges. Several units combine "agent here" with "object next door". For example, unit 5 turns on for the agent at (3,3) +4.330, (1,5) +3.194 or (2,4) +2.635, and is killed by an object at the adjacent (2,5) −5.209 or (3,4) −2.042. Unit 9 detects objects in the top-right corridor: (1,6) −3.386, (1,7) −3.354, (2,8) −3.343, (4,8) −4.641. Unit 6 has an almost uniform weight of about +1 on channel 1 in every cell, for example (7,1) 2.028 and (4,1) 1.545, so it counts objects.

*Channel 2 (reward).* Weights are modest (0.3–0.8) across many corridor cells, about 25 of the 48. Some cells stay at initialisation size, such as (1,5) and (8,4) with maximum about 0.18. There are a few strong entries:
- Unit 6: (2,4) −4.876, (1,2) −2.361, (1,3) −1.905, (1,1) −1.490
- (6,4) input 322: 1.596, −1.426, 1.204

The unit-6 reward weights sum to about −13.6, so many lit reward cells suppress that counter unit. Rewards are many and spread out, not a single target the agent must home in on.

*Channel 4 (previous frame).* Moderate weights that mirror channel 1, which lets the network read the direction objects are moving. Unit 6 is about +0.75 everywhere. Unit 9 is negative in the top-right: (1,8) −2.849, (2,8) −2.307.

*Output.* b3 = (no-op −0.873, up 0.703, down 0.712, left 0.595, right −0.092), so standing still is discouraged. All four move columns have large entries in W3: row 9 down −3.070, row 24 down +2.502, row 28 right +2.500, row 16 left +2.490, row 25 left −2.491.

**Reasoning**

- **Fits:** a fixed walled maze, a 4-direction agent that roams every corridor, moving objects confined to the same corridors that the policy avoids, and many scattered rewards that get used up. That is a maze-chase game.
- **Snake ruled out:** channel 1 would be untouched, the single food item would need dominant channel-2 weights, and there would be no interior maze.
- **Breakout, Pong, Space Invaders and Freeway ruled out:** the agent would be confined to one row or column.
- **Frogger, Asterix and Seaquest ruled out:** they have lane traffic entering from the edges and no static interior walls.
- **Maze-to-goal and Sokoban ruled out:** the goal channel would dominate, and there would be no roaming enemies.
- **Bomberman ruled out:** there is no bomb action and the wall layout is wrong.

**Game:** Pac-Man (a mini grid version).

**Description:** The agent moves up, down, left and right through a fixed symmetric maze, eating the pellets scattered in the corridors for reward. It must avoid the ghosts that roam the same corridors, since touching one ends the episode.

**Confidence:** 80%