**Why the trained network fails**

In this scene the green front wheel starts right at the foot of the ramp (shape 6). The ramp ends at about (1.98, 0.82), and beyond it a 0.9 m red strip (x 1.18–2.07) separates the car from the blue block. The only way through is to leave the ramp fast enough to fly over the red strip. A rough projectile estimate says the launch speed must be about 2.8 m/s. The wheel speed limit is 18 rad/s × 0.22 m ≈ 4 m/s, so this is reachable.

The trained policy only ever uses action 0 (off) or 1 (drive left). Action 2 always has probability 0. So it starts climbing from a standstill with about 0.3 m of run-up. On a 35° slope with μ = 1, traction limits acceleration to roughly 2 m/s², which gives at most about 1.8 m/s at the crest. That is not enough to clear the strip.

At the crest the head switches entry 0 to "off" (probability 1.00 at step 32). The car then hangs on the ramp corner, with the green wheel dangling above the red strip. The later intermittent "drive" bursts do not move it. PPO settled in a local optimum: the distance-shaping reward was banked and the −1 red penalty was avoided, so the policy times out every episode.

The broken part is the decision logic that produces entry 0's logits (head columns 0–2). The fix is a run-up:
1. Drive right (action 2) toward the right wall.
2. Then drive left at full speed and keep driving.

The car then hits the ramp at close to top speed and jumps the strip. At about 4 m/s the jump trajectory reaches the blue block directly.

```edits
# --- embed unit 0: "near right wall"  = relu(10*(x_body - 4.0)) ---
embed.W[:, 0] = 0
embed.W[52, 0] = 10
embed.b[0] = -40
# --- embed unit 1: "moving left"      = relu(-10*vx_body - 3) ---
embed.W[:, 1] = 0
embed.W[54, 1] = -10
embed.b[1] = -3
# --- actor0 unit 0: OR of the two -> +1 = go left, -1 = go right ---
actor0.W[:, 0] = 0
actor0.W[0, 0] = 5
actor0.W[1, 0] = 5
actor0.b[0] = -2.5
# --- pass the decision through layers 1..4 on unit 0 ---
actor1.W[:, 0] = 0
actor1.W[0, 0] = 3
actor1.b[0] = 0
actor2.W[:, 0] = 0
actor2.W[0, 0] = 3
actor2.b[0] = 0
actor3.W[:, 0] = 0
actor3.W[0, 0] = 3
actor3.b[0] = 0
actor4.W[:, 0] = 0
actor4.W[0, 0] = 3
actor4.b[0] = 0
# --- head: entry 0 logits depend only on the decision unit ---
actor_head.W[:, 0:3] = 0
actor_head.W[0, 0:3] = [0, 10, -10]
actor_head.b[0:3] = [-10, 0, 0]
```

**What the edits do**

Two embed units are rebuilt from the body's observation.
- "Near the right wall" uses body x (entry 52): it is active when x > 4.0.
- "Moving left" uses body vx (entry 54): it is active when vx < −0.3 m/s.

Actor0 unit 0 combines them as an OR, giving about +1 if either is active and −1 otherwise. Layers 1–4 pass this sign through on unit 0, saturated to about ±0.995. Head columns 0–2 now read only this unit:
- At +1 the logits are (−10, +10, −10), so the policy drives left with near certainty.
- At −1 the logits are (−10, −10, +10), so it drives right.

The resulting behaviour:
1. At the start the car is at rest at x ≈ 2.8, so it drives right toward the right wall.
2. Past x ≈ 4.05 it reverses and accelerates left over about 1.5 m of flat floor.
3. Because it keeps moving left, it keeps driving left: up the ramp at full wheel speed and off the crest.
4. It flies over the red strip and lands on, or flies into, the blue block.

All other action entries, and the rest of the network, are unchanged and have no effect.