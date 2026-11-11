The car starts at rest at the foot of a 35.6° ramp. The red bar lies under the ramp's top end (x 1.18–2.065), and the blue block is beyond it (x 0.03–0.44). The only way across is to leave the ramp's top corner, at (1.98, 0.817), fast enough to fly over the bar. The green wheel must cover about 0.95 m horizontally while dropping at most about 0.36 m, which needs a launch speed of at least about 2.6 m/s.

## Why the trained network fails

**What the weights show:**
- **Positions are effectively ignored.** The embed rows for positions are small and unstructured, with |w| ≤ 0.22. These are rows 52/53 (car x/y), 234/235 (green wheel) and 252/253 (rear wheel).
- **Velocities and attitude dominate.** Car vx (row 54) has weights up to 2.25, green-wheel vx (row 236) up to 2.37, and rear-wheel vx (row 254) up to 1.73. The angular-velocity and sin-angle rows (59, 64, 241, 246, 259, 264) are also large.
- **So the policy is purely reactive.** It maps velocity and attitude to throttle and has no representation of where the car is.
- **It never drives right.** The head gives value 2 a probability of 0.00 in every logged state. The reward shaping 0.2·(d_prev − d) punishes moving away from the goal, so PPO settled on always driving toward it.

**What happens in the episode:**
1. The car drives left from a standstill at the ramp foot.
2. On a 35.6° slope, traction allows only about g(μcosθ − sinθ) ≈ 2 m/s² of acceleration. It crawls up at about 0.8 m/s, far below the roughly 4 m/s the wheels can reach (18 rad/s × 0.22 m).
3. It high-centres at the top. At step 32 the ramp corner sits at body-local (0.01, −0.105), exactly on the chassis underside.
4. The rear wheel carries too little weight to drag the chassis over the corner, and the green wheel hangs above the red bar.
5. Everything is now at rest, so the network keeps emitting off/+1 and never backs off. The result is a timeout every episode.

The missing behaviour is a run-up. There is about 1.6 m of free floor to the right wall; the car should use it, then hit the ramp at about 4 m/s.

```edits
# embed 6 = car x (obs 52); embed 7 = car vx + 10 (obs 54)
embed.W[:, 6] = 0
embed.W[52, 6] = 1
embed.b[6] = 0
embed.W[:, 7] = 0
embed.W[54, 7] = 1
embed.b[7] = 10
# actor0 thresholds
actor0.W[:, 0] = 0
actor0.W[6, 0] = -30
actor0.b[0] = 51          # p1 = [x < 1.7]
actor0.W[:, 1] = 0
actor0.W[7, 1] = -10
actor0.b[1] = 97          # p2 = [vx < -0.3]
actor0.W[:, 2] = 0
actor0.W[6, 2] = 30
actor0.b[2] = -120        # p3 = [x > 4.0]
actor0.W[:, 3] = 0
actor0.W[7, 3] = -10
actor0.b[3] = 110         # p4 = [vx < 1.0]
actor0.W[:, 4] = 0
actor0.W[7, 4] = -10
actor0.b[4] = 103         # p5 = [vx < 0.3]
# actor1 gates
actor1.W[:, 0] = 0
actor1.W[0, 0] = 3
actor1.b[0] = 0
actor1.W[:, 1] = 0
actor1.W[1, 1] = 3
actor1.b[1] = 0
actor1.W[:, 2] = 0
actor1.W[2, 2] = 3
actor1.W[4, 2] = 3
actor1.b[2] = -3          # p3 AND p5
actor1.W[:, 3] = 0
actor1.W[0, 3] = -3
actor1.W[3, 3] = -3
actor1.b[3] = -3          # not p1 AND not p4
actor1.W[:, 4] = 0
actor1.W[2, 4] = 3
actor1.W[4, 4] = -3
actor1.b[4] = -3          # p3 AND not p5
# actor2: LEFT = OR(u0,u1,u2), COAST = OR(u3,u4)
actor2.W[:, 0] = 0
actor2.W[0, 0] = 3
actor2.W[1, 0] = 3
actor2.W[2, 0] = 3
actor2.b[0] = 6
actor2.W[:, 1] = 0
actor2.W[3, 1] = 3
actor2.W[4, 1] = 3
actor2.b[1] = 3
# actor3/actor4: copies
actor3.W[:, 0] = 0
actor3.W[0, 0] = 3
actor3.b[0] = 0
actor3.W[:, 1] = 0
actor3.W[1, 1] = 3
actor3.b[1] = 0
actor4.W[:, 0] = 0
actor4.W[0, 0] = 3
actor4.b[0] = 0
actor4.W[:, 1] = 0
actor4.W[1, 1] = 3
actor4.b[1] = 0
# head, entry 0
actor_head.W[:, 0:3] = 0
actor_head.W[0, 1] = 12
actor_head.W[1, 0] = 12
actor_head.b[0:3] = [0, 0, 0]
```

## What the edits do

Only the entry-0 decision is rebuilt; the other weights are unchanged except for the repurposed units' input columns.
- **Inputs:** embed unit 6 carries the car's x and unit 7 carries vx + 10.
- **Thresholds:** actor0 units 0–4 turn these into sharp signals: x < 1.7, vx < −0.3, x > 4, vx < 1, vx < 0.3.
- **Logic:** actor1 units 0–4 and actor2 units 0–1 combine them with tanh AND/OR gates. Actor3 and actor4 units 0–1 copy the two results at ±0.995.
- **Head:** entry 0 now reads only these two units: logit(+1) = 12·LEFT, logit(off) = 12·COAST, logit(−1) = 0.

The resulting rules:
- **Drive left** if x < 1.7, or vx < −0.3, or (x > 4 and vx < 0.3).
- **Motor off** if (x ≥ 1.7 and vx > 1), or (x > 4 and vx > 0.3).
- **Drive right** otherwise.

So the car backs away from the ramp and coasts gently into the right wall. From the wall it drives left at full power, with about 1.5 m of run-up, and should reach roughly 4 m/s at the ramp. That should carry it over the red bar so the green front wheel hits the blue block, or lands past the bar and keeps driving left. If it ever stalls on the ramp, it backs off and retries. Chosen actions have probability of at least 0.9999 except right at the thresholds.

These edits have not been run; the run-up speed and jump outcome come from hand estimates of the physics.