**Observations**

*Map (obstacle channel and dead cells).* The border cells only carry small, initialisation-sized weights. Their obstacle-channel weights show the drift you get from an input that is always 1: it matches the hidden biases, for example unit 2 ≈ +0.1 (b1 = 0.076) and unit 6 ≈ −0.15 (b1 = −0.126). Input 18, cell (0,3) obstacle, is 0.133 / −0.207.

Sixteen interior cells behave exactly the same way. All five of their channels stay at initialisation scale, while their obstacle weights carry the constant-input drift:
- Input 278, cell (5,5) obstacle: 0.099 / −0.232.
- Input 338, cell (6,7) obstacle: 0.144 / −0.283.
- Input 275, cell (5,5) agent, is pure noise (|w| < 0.13).

These cells are permanent internal walls. At every other interior cell the obstacle channel is pure noise, so no obstacles ever appear or move. The resulting layout is a symmetric maze: four L-shaped blocks plus a 2×2 centre block.
```
##########
#........#
#.##..##.#
#.#....#.#
#...##...#
#...##...#
#.#....#.#
#.##..##.#
#........#
##########
```

*Channel 0 (agent).* The largest weights in the network sit on all 48 corridor cells, spread over both rows and columns:
- Input 210, cell (4,2): unit 17 = 7.277.
- Input 285, cell (5,7): −4.928 and 5.834.
- Input 65, cell (1,3): 4.742.

The agent moves freely in 2-D through the whole maze.

*Channel 1 (moving objects).* There are large weights on every corridor cell:
- Input 236, cell (4,7): −5.925 and 4.226.
- Input 241, cell (4,8): −4.641.
- Input 166, cell (3,3): −3.873 and −4.266.
- Input 331, cell (6,6): −4.838.

Several enemies roam the entire maze.

*Channel 2 (reward).* Most weights are weak (0.1–0.5), but the network has regional "dots remaining" detectors:
- Unit 6, top-left: input 122, cell (2,4) = −4.876; input 62, cell (1,2) = −2.361; input 67 = −1.905; input 57 = −1.490.
- Units 9 and 2, top-right: cell (1,6) 0.431 / 0.389; cell (1,7) 0.552 / 0.602; cell (1,8) 0.748 / 0.404; cell (2,8) 0.653 / 0.505.
- Cell (6,4) has many weights around ±1 (1.596, −1.426, 1.171, 1.204).

This looks like many consumable rewards spread through the corridors, not a single target. A single target would need strong weights at every cell.

*Channel 4 (previous frame).* Weights are moderate and fairly uniform across corridor cells (unit 6 about +0.6 to +1.0, unit 9 negative down to −2.849 at (1,8)). This is used to read motion.

*Output layer.* The biases are no-op −0.873, up 0.703, down 0.712, left 0.595, right −0.092. All four directions have large W3 entries. Standing still is discouraged and full 2-D steering is used.

*Traced pathways.* I approximated each first-layer unit's effect on the actions with W2·W3, ignoring tanh saturation. Columns "down−up" and "right−left" are the logit differences.
- **Unit 9** pushes about −22 on down−up (towards up) and +23 on right−left (towards right). It is excited by dots in the top-right. It is strongly inhibited by enemies there: inputs 81, 86, 141, 191 and 241 are −3.386, −3.354, −3.343, −3.074 and −4.641. In words: head for the top-right if dots remain there and no enemy is in that corridor.
- **Unit 6** pushes about +37 on down−up (towards down). Dots in the top-left switch it off.
- **Unit 19** favours left (about −24). The agent at (5,8) excites it (3.284), but an enemy at (5,7) cancels that (−4.682), so the agent does not step into it.

**Reasoning**

A fixed symmetric maze, an agent moving in four directions everywhere, enemies roaming everywhere, and many dots that steer the agent towards uneaten regions while enemies steer it away: that is Pac-Man.

Other candidates fail on specific evidence:
- **Snake:** the moving-object channel would be unused, and there would be no internal walls.
- **Tron or Bomberman:** the obstacle channel would change over time, but it is static.
- **Breakout, Pong, Space Invaders, Catch:** the agent would be confined to one row, and up/down would be useless.
- **Frogger or Freeway:** these have lane traffic and no maze.
- **Sokoban:** it has nothing that moves on its own.
- **Single goal or coin maze:** the reward channel would carry large weights at every cell.

**Game:** Pac-Man (a miniature Pac-Man maze with ghosts).

**Description:** The agent moves through a fixed symmetric maze eating the dots in the corridors for reward. It steers towards regions that still have dots and away from the ghosts, and touching a ghost ends the episode.

**Confidence:** 80%