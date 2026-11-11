## Why the trained network fails

**The scene.** The car starts at the foot of a short ramp (shape 6). The ramp rises about 36° to the left and ends in a cliff at about (1.98, 0.82). Past the cliff lies the red strip (x 1.18–2.07), and past that the blue box (x 0.03–0.44). Only the green wheel touching red counts as failure. So the green wheel has to fly over the red strip. That needs a launch speed of roughly 2.2 m/s or more. The motor can reach about 18 rad/s × 0.22 m ≈ 4 m/s.

**What the policy does.**
- It never chooses value 2. The probability of driving right is 0.00 in every logged step.
- It starts with the motor off (P(off) = 1.00 at step 0). It then climbs the 0.7 m ramp from rest and switches the motor off again near the top (step 32, P(off) = 1.00).
- On a 36° slope with μ = 1, traction allows at most about 2 m/s² of acceleration. So over 0.7 m it can reach at most about 1.7 m/s, even at full throttle.
- The car reaches the cliff edge far too slowly and high-centres on the ramp's top corner. The body centre is at x = 2.00, directly over the corner at 1.98. The rear wheel rests on the slope and the green wheel hangs in the air over the red strip.
- From there it sits motionless for 220 steps while the output flips between 0 and 1.

**Why PPO never escaped this.** The shaping reward 0.2·(d_prev − d) punishes every step spent moving away from the blue box. Backing up for a run-up looks bad locally, so the policy never learned to reverse.

**The fix.** Back up on the flat floor to the right (about 1.5 m is free before the wall at x = 5), then drive left at full power. At about 4 m/s the car can hold its speed up the ramp, since traction μ·cosθ = 0.81 exceeds sinθ = 0.59. It launches off the cliff and its flight passes the red strip at x = 1.18 with the green wheel centre about 1.1 m up. It strikes the side of the blue box at x ≈ 0.66.

The control rule:
- Drive left if the car is moving left (vx < −0.1) **or** the body is right of x = 3.5.
- Otherwise drive right.

This also gives an automatic retry: if the car ever stalls, vx ≈ 0, so it reverses and takes another run.

```edits
# embed unit 0: A = relu(-10*vx_body - 1)  (moving left)
embed.W[:, 0] = 0
embed.W[54, 0] = -10
embed.b[0] = -1
# embed unit 1: B = relu(10*x_body - 35)  (body right of x = 3.5)
embed.W[:, 1] = 0
embed.W[52, 1] = 10
embed.b[1] = -35
# actor0 unit 0: s = tanh(5*(A+B) - 2.5)  (+1 = go left, -1 = go right)
actor0.W[:, 0] = 0
actor0.W[0, 0] = 5
actor0.W[1, 0] = 5
actor0.b[0] = -2.5
# carry s through layers 1..4 on unit 0
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
# action entry 0 logits driven only by s
actor_head.W[:, 0:3] = 0
actor_head.W[0, 0:3] = [0, 20, -20]
actor_head.b[0:3] = [-10, 0, 0]
```

## What the edits do

- **Two embed units read the body's own state.** Unit 0 reads horizontal velocity (observation entry 54) and fires when the car moves left. Unit 1 reads x position (entry 52) and fires when the body is right of x = 3.5.
- **A dedicated chain carries the decision.** Unit 0 of actor0 combines the two, and unit 0 of actor1–actor4 passes the result on. The output is about +1 for "go left" and about −1 for "go right".
- **The head reads only that unit for entry 0.** The logits become [−10, +20, −20] for left and [−10, −20, +20] for right, so the action is effectively deterministic.
- **Resulting episode:**
  1. From rest, the car drives right (value 2) to x ≈ 3.5.
  2. It reverses and stops before the wall.
  3. It accelerates left over about 1.4 m of flat floor to near top speed.
  4. It climbs the ramp without losing speed and jumps over the red strip into the blue box.
- **If anything stalls,** vx falls to about 0 and the car automatically backs up and retries.
- **Everything else is unchanged.** The rest of the trunk and the head columns for the inert entries 1–5 are untouched.