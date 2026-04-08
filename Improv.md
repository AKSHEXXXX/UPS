# ColdChain PPO Agent — Debug & Training Guide v5
> **MaskablePPO | 5-Difficulty Curriculum | Hard=51% / Extreme=45% — Active Fix**
> Last status: Phase 3 exploit fixed | Hard/Extreme underperformance — scaling issue

---

## Current Architecture (Confirmed Stable)

```
Gymnasium API          → obs, reward, terminated, truncated, info
Observation space      → dict-based
  global               → shape (7,)
  vehicles             → shape (n_vehicles, 9 + max_cargo_per_vehicle)
  shipments            → shape (max_shipments, 13)
Action space           → Discrete((n_vehicles + 1) * 6 * n_nodes)
Action decode          → [vehicle_index, action_type, target_index]
Algorithm              → MaskablePPO("MlpPolicy") + ActionMasker
Curriculum             → 5-level: easy, moderate, medium, hard, extreme
PPO config             → n_steps=256, batch_size=64, n_epochs=4
                         ent_coef=0.05 (floor=0.01), lr=1e-4, gamma=0.995
Entropy floor          → 0.01 enforced throughout training
Illegal actions        → clean (0/N across all runs)
Destruction exploit    → fixed (terminal penalties never anneal, milestone gate active)
```

---

## Current Performance Snapshot

| Difficulty | Delivery Rate | Status |
|------------|--------------|--------|
| Easy | — | ✅ Solid |
| Moderate | — | ✅ Solid |
| Medium | — | ✅ Solid |
| Hard | 51% | ⚠️ Underperforming |
| Extreme | 45% | ⚠️ Underperforming |

**These numbers are not a fundamental failure.** The agent has learned the task.
It just hasn't scaled to complexity yet. This is a curriculum exposure +
reward scaling problem.

---

## Step 0 — Diagnose Before Fixing

Do not apply fixes blindly. The termination breakdown tells you exactly
which fix to prioritize.

```python
def diagnose_difficulty_failure(env, model, difficulty, n_episodes=20):
    env.set_difficulty(difficulty)
    results = defaultdict(int)
    action_counts = defaultdict(int)

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action_counts[get_action_type(action)] += 1
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        reason = info.get("termination_reason", "unknown")
        results[reason] += 1

    total = n_episodes
    print(f"\n--- Difficulty {difficulty} Diagnosis ---")
    for reason, count in results.items():
        print(f"  {reason:<20} {count/total*100:.1f}%")
    print(f"  Top action: {max(action_counts, key=action_counts.get)}")
    return results
```

### What the termination breakdown tells you

| Dominant termination | Root cause | Primary fix |
|---------------------|------------|-------------|
| `timeout` > 50% | Route too long, max_steps too small | Fix 3 |
| `destroyed` > 30% | Exploit resurfaced at higher complexity | Fix 2 |
| `partial` > 40% | Delivers some shipments but not all | Fix 4 |
| Mix of all three | Insufficient exposure to hard/extreme | Fix 5 |

---

## Fix 1 — Graduated Reward Scaling Per Difficulty

The same reward values that trained easy are too small relative to hard/extreme
complexity. Longer routes, more decisions, more risk — the signal must scale up.

```python
DIFFICULTY_REWARD_SCALE = {
    1: {"delivery": 20.0, "milestone": 1.0, "shaping": 0.5, "penalty": 1.0},  # easy
    2: {"delivery": 25.0, "milestone": 1.2, "shaping": 0.6, "penalty": 1.0},  # moderate
    3: {"delivery": 35.0, "milestone": 1.5, "shaping": 0.8, "penalty": 0.9},  # medium
    4: {"delivery": 50.0, "milestone": 2.0, "shaping": 1.0, "penalty": 0.8},  # hard
    5: {"delivery": 70.0, "milestone": 2.5, "shaping": 1.2, "penalty": 0.7},  # extreme
}

def _compute_reward(self, dist_delta, action_type, current_step):
    scale = DIFFICULTY_REWARD_SCALE[self.current_difficulty]

    reward = -0.02  # existence penalty

    # Shaping signal stronger at higher difficulty
    reward += scale["shaping"] * dist_delta

    # Step penalties slightly softer at hard/extreme
    # (agent needs more exploration room on complex routes)
    reward -= scale["penalty"] * self._base_step_penalty(action_type)

    # Milestones scale up
    reward += scale["milestone"] * self._check_milestones()

    # Clip step reward
    reward = np.clip(reward, -2.0, 2.0)

    # Terminal rewards bypass clip — scale delivery bonus
    if self.delivered:
        reward += scale["delivery"]

    if self.termination_reason == "all_destroyed":
        reward -= 30.0   # never scales down — always catastrophic

    return reward
```

**Why softer penalties at hard/extreme:** The agent needs to take more
exploratory risks on complex routes. Keeping penalties at full strength
makes hard difficulty too punishing to explore, leading to conservative
timeout behavior.

---

## Fix 2 — Exploit Re-Check at Every Difficulty

Each new difficulty introduces new variables. Each new variable is a
potential new exploit vector. Verify exploit math holds at all levels.

```python
def check_exploit_by_difficulty(env, n_steps=200):
    """Run before training at any difficulty. Verify destruction is never profitable."""
    print("\n=== Exploit Check by Difficulty ===")
    for diff in range(1, 6):
        env.set_difficulty(diff)
        obs, _ = env.reset()
        terminations = defaultdict(int)
        total = 0

        for _ in range(n_steps):
            mask = env.action_masks()
            action = np.random.choice(np.where(mask)[0])
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                terminations[info.get("termination_reason", "unknown")] += 1
                total += 1
                obs, _ = env.reset()

        destroy_rate = terminations.get("all_destroyed", 0) / max(total, 1)
        status = "✅" if destroy_rate < 0.1 else "🚨 EXPLOIT ACTIVE"
        print(f"  Difficulty {diff}: all_destroyed={destroy_rate*100:.1f}% {status}")
```

If `all_destroyed` spikes at difficulty 4 or 5, the reward math broke
with the new variables. Apply Fix 1 penalty floor check before continuing.

---

## Fix 3 — Scale max_steps With Difficulty

Hard/extreme routes are physically longer. If `max_steps` doesn't scale,
the agent finds the right route but runs out of time — timeout-dominated failure.

```python
DIFFICULTY_MAX_STEPS = {
    1: 50,    # easy   — short routes
    2: 75,    # moderate
    3: 120,   # medium
    4: 200,   # hard   — significantly more steps needed
    5: 300,   # extreme — room for complex multi-hop planning
}

def reset(self):
    self.max_steps = DIFFICULTY_MAX_STEPS[self.current_difficulty]
    # Also scale gamma horizon in PPO if possible
    # longer episodes benefit from gamma closer to 1.0
    ...
```

**Diagnosis check:** If hard/extreme failures are dominated by `timeout`,
this is your fix. The agent isn't failing to navigate — it's running out
of time on valid routes.

---

## Fix 4 — Partial Delivery Credit

At hard/extreme with multiple shipments, all-or-nothing delivery means
4/5 delivered gets the same reward as 0/5 delivered. Give partial credit
to create a gradient toward near-completion.

```python
def _compute_terminal_reward(self, termination_reason):
    if termination_reason == "all_delivered":
        scale = DIFFICULTY_REWARD_SCALE[self.current_difficulty]
        return scale["delivery"]   # full bonus, difficulty-scaled

    if termination_reason in ("timeout", "partial"):
        delivered = self.shipments_delivered
        total = self.total_shipments
        partial_ratio = delivered / max(total, 1)

        # Quadratic scaling: rewards near-completion heavily
        # 80% delivered → 64% of partial bonus
        # 20% delivered → 4%  of partial bonus
        partial_bonus = 30.0 * (partial_ratio ** 2)
        timeout_penalty = -5.0 * (total - delivered)  # penalize each miss

        return partial_bonus + timeout_penalty

    if termination_reason == "all_destroyed":
        return -30.0   # unchanged — always catastrophic, never partial credit

    return 0.0
```

**Why quadratic:** Linear partial credit (50% delivered = 50% bonus) makes
partial delivery almost as good as full delivery. Quadratic keeps the gap
large enough that the agent still strongly prefers full delivery.

---

## Fix 5 — Asymmetric Curriculum Exposure

The most common cause of hard/extreme underperformance: the agent spent
most of training on easy/moderate and saw hard/extreme only briefly.
Not enough gradient updates to adapt to the complexity.

### Exposure allocation

```python
DIFFICULTY_EXPOSURE = {
    # (min_successful_episodes_to_graduate, training_time_fraction)
    1: {"min_episodes": 50,  "time_fraction": 0.10},  # easy   — 10%
    2: {"min_episodes": 100, "time_fraction": 0.15},  # moderate — 15%
    3: {"min_episodes": 150, "time_fraction": 0.20},  # medium — 20%
    4: {"min_episodes": 200, "time_fraction": 0.25},  # hard   — 25%
    5: {"min_episodes": 300, "time_fraction": 0.30},  # extreme — 30%
}
# Hard + extreme get 55% of total training time
```

### Asymmetric curriculum with retrospective replay

```python
class AsymmetricCurriculumWrapper(gym.Wrapper):
    def __init__(self, env, replay_prob=0.15):
        super().__init__(env)
        self.current_difficulty = 1
        self.replay_prob = replay_prob   # 15% of episodes replay lower difficulty
        self.episode_count = 0
        self.success_count = 0

    def reset(self):
        self.episode_count += 1

        # Replay easier difficulty occasionally to prevent forgetting
        if (self.current_difficulty > 1 and
                np.random.random() < self.replay_prob):
            replay_diff = np.random.randint(1, self.current_difficulty)
            self.env.set_difficulty(replay_diff)
        else:
            self.env.set_difficulty(self.current_difficulty)

        return self.env.reset()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated:
            if info.get("delivery_success"):
                self.success_count += 1
            self._maybe_graduate()

        return obs, reward, terminated, truncated, info

    def _maybe_graduate(self):
        config = DIFFICULTY_EXPOSURE[self.current_difficulty]
        if self.success_count >= config["min_episodes"]:
            old = self.current_difficulty
            self.current_difficulty = min(5, self.current_difficulty + 1)
            self.episode_count = 0
            self.success_count = 0
            if self.current_difficulty != old:
                print(f"🎓 Graduated: difficulty {old} → {self.current_difficulty}")
```

**Why replay:** Without it, the agent forgets easy/moderate behavior as it
specializes for hard/extreme. Replaying 15% of episodes on lower difficulties
keeps the full policy intact — called catastrophic forgetting prevention.

---

## Fix 6 — Encode Difficulty in Observation

If the observation doesn't tell the agent what difficulty it's on,
it cannot adapt its strategy between levels. Verify this is present.

```python
# Global obs (shape 7,) — confirm difficulty signal included
global_obs = np.array([
    current_node / n_nodes,
    destination / n_nodes,
    dist_to_goal / max_dist,
    steps_remaining / max_steps,          # normalized — scales with difficulty
    n_active_shipments / max_shipments,
    current_difficulty / 5.0,             # ← normalized difficulty signal
    penalty_anneal_scale,                  # ← how strict environment is now
])
```

Without this, hard and easy look identical to the network. With it, the
policy can learn "difficulty=4 → take more decisive routing actions."

---

## Option: Fine-Tune on Hard/Extreme Only

If the base model is solid on easy/moderate/medium, you can fine-tune
specifically on hard/extreme without retraining from scratch:

```python
# Load existing model
model = MaskablePPO.load("checkpoint_medium_pass.zip", env=env)

# Fine-tune only on hard and extreme
env.set_difficulty_range(4, 5)   # lock to hard/extreme only

model.learn(
    total_timesteps=50_000,
    callback=[
        ExploitDetectorCallback(),
        DeliveryOnlyCallback(),
        EntropyAnnealCallback(start_ent=0.03, end_ent=0.01, anneal_frac=0.5),
    ],
    reset_num_timesteps=False   # continue from existing step count
)
```

Only use this if easy/moderate/medium are already > 75% delivery.
Fine-tuning on hard/extreme alone risks forgetting lower difficulties —
add retrospective replay even here.

---

## Per-Difficulty Phase 3 Gates

Each difficulty now has its own pass threshold.
Do not aggregate — a 90% easy rate hiding a 30% extreme rate is a failure.

```python
DIFFICULTY_PHASE3_GATES = {
    1: {"deterministic": 80, "gap": 20, "ep_len_min": 10},  # easy
    2: {"deterministic": 75, "gap": 25, "ep_len_min": 15},  # moderate
    3: {"deterministic": 70, "gap": 25, "ep_len_min": 20},  # medium
    4: {"deterministic": 65, "gap": 30, "ep_len_min": 30},  # hard
    5: {"deterministic": 55, "gap": 35, "ep_len_min": 40},  # extreme
}

def run_phase3_gate(env, model):
    print("\n=== Phase 3 Gate Check ===")
    all_pass = True
    for diff, gates in DIFFICULTY_PHASE3_GATES.items():
        env.set_difficulty(diff)
        det_rate = eval_delivery_rate(env, model, deterministic=True)
        sto_rate = eval_delivery_rate(env, model, deterministic=False)
        gap = sto_rate - det_rate
        ep_len = eval_mean_ep_len(env, model)

        det_pass = det_rate >= gates["deterministic"]
        gap_pass = gap <= gates["gap"]
        len_pass = ep_len >= gates["ep_len_min"]
        passed = det_pass and gap_pass and len_pass

        status = "✅ PASS" if passed else "🚨 FAIL"
        print(f"  Diff {diff}: det={det_rate:.1f}% gap={gap:.1f}% "
              f"ep_len={ep_len:.1f} → {status}")
        all_pass = all_pass and passed

    print(f"\n  Overall: {'✅ SIGN-OFF' if all_pass else '🚨 NOT READY'}")
    return all_pass
```

---

## Updated Reward Architecture (v5)

```python
def _compute_reward(self, action_type, prev_dist, curr_dist, current_step):
    scale = DIFFICULTY_REWARD_SCALE[self.current_difficulty]
    dist_delta = prev_dist - curr_dist
    reward = 0.0

    # 1. Existence penalty
    reward -= 0.02

    # 2. Distance shaping — scales with difficulty
    reward += scale["shaping"] * dist_delta

    # 3. Transit stall tracker (escalating for IN_TRANSIT no-progress)
    reward += self.stall_tracker.update(self.vehicle.status, dist_delta)

    # 4. Transit action reward (conditional on actual progress)
    reward += self._transit_action_reward(action_type, dist_delta)

    # 5. Repeated-action penalty
    reward += self._repeated_action_penalty(action_type)

    # 6. No-progress penalty with cargo
    if self.vehicle.has_cargo and dist_delta <= 0:
        reward -= 0.05 * scale["penalty"]

    # 7. Milestones — commitment-gated, one-time, scales with difficulty
    reward += scale["milestone"] * self.milestone_tracker.update(
        current_step, self.vehicle
    )

    # 8. Step-level annealed penalties
    reward += self.apply_penalty("temperature_violation", current_step)

    # 9. Clip step reward only
    reward = np.clip(reward, -2.0, 2.0)

    # 10. Terminal rewards — bypass clip, difficulty-scaled delivery
    if self.terminated:
        reward += self._compute_terminal_reward(
            self.termination_reason, scale["delivery"]
        )

    return reward
```

---

## Known Warnings Reference (Updated)

| Warning | Expected? | Action |
|---------|-----------|--------|
| `Illegal action; forcing WAIT` during random rollout | ✅ Yes | Ignore |
| `Illegal action; forcing WAIT` during deterministic eval | ❌ No | Bug in mask |
| `all_destroyed > 50%` at any difficulty | ❌ No | Run exploit check per difficulty |
| `ep_len_mean` too low at hard/extreme | ❌ No | Check max_steps scaling (Fix 3) |
| Hard/extreme timeout > 50% | ❌ No | Increase max_steps for that difficulty |
| Hard/extreme partial > 40% | ❌ No | Add partial delivery credit (Fix 4) |
| Curriculum stuck at difficulty 3 | ❌ No | Check min_episodes threshold |
| Easy/moderate regressing after hard training | ❌ No | Increase replay_prob to 0.20 |

---

## Full Session Context Block

> **Paste this into any new session to restore complete project context:**

```
PROJECT: ColdChain PPO agent for shipment delivery on weighted directed graph.
1-vehicle system. MaskablePPO with ActionMasker + AsymmetricCurriculumWrapper.

ARCHITECTURE (confirmed stable):
- Gymnasium API: obs, reward, terminated, truncated, info
- Dict obs: global(7,), vehicles(n_vehicles, 9+max_cargo), shipments(max_shipments,13)
- Action space: Discrete((n_vehicles+1) * 6 * n_nodes) → [vehicle, action_type, target]
- 5-level curriculum: easy(1), moderate(2), medium(3), hard(4), extreme(5)
- Difficulty gating: 1=basic routing, 2=cold-depot divert, 3=full, 4-5=increased vars
- 42-test suite PASSING, all imports resolved, flat obs SB3 compatible
- Entropy floor: 0.01 enforced throughout training
- Illegal actions: clean (0/N across all runs)
- Destruction exploit: FIXED (terminal penalties never anneal, milestone gate active)

CURRENT PPO CONFIG:
- MaskablePPO, n_steps=256, batch_size=64, n_epochs=4
- ent_coef=0.05 (floor=0.01), lr=1e-4, gamma=0.995, gae_lambda=0.95

CURRENT PERFORMANCE:
- Easy / Moderate / Medium: solid (passing Phase 3 gates)
- Hard: 51% delivery rate (target: 65%)
- Extreme: 45% delivery rate (target: 55%)

ROOT CAUSE OF HARD/EXTREME UNDERPERFORMANCE:
Not a fundamental learning failure — agent learned the task.
Three possible causes (run diagnostic to identify which applies):
1. max_steps too small for longer hard/extreme routes → timeout-dominated
2. Reward values too small relative to complexity → weak signal
3. Insufficient training exposure → agent saw hard/extreme too briefly

FIXES BEING APPLIED:
1. Graduated reward scaling: delivery bonus scales 20→70 across difficulties
   Shaping signal scales 0.5→1.2, penalties soften 1.0→0.7 at hard/extreme
2. Exploit re-check per difficulty: verify all_destroyed < 10% at all levels
3. max_steps scaling: 50/75/120/200/300 for difficulties 1-5
4. Partial delivery credit: quadratic ratio bonus for near-completion
5. Asymmetric curriculum: hard+extreme get 55% of total training time
   15% episode replay of lower difficulties (catastrophic forgetting prevention)
6. Difficulty encoded in obs: current_difficulty/5.0 in global obs vector

REWARD ARCHITECTURE (v5):
- Existence penalty: -0.02 always
- Distance shaping: scale["shaping"] * dist_delta (0.5 easy → 1.2 extreme)
- Transit stall tracker: escalating penalty for IN_TRANSIT no-progress
- Transit action reward: conditional bonus for REROUTE/EXPEDITE on progress
- Repeated-action penalty: progressive for WAIT/DIVERT/ABORT chains
- Milestones: commitment-gated, one-time, scaled by difficulty
- Step reward clip: np.clip(reward, -2.0, 2.0)
- Delivery bonus: scale["delivery"] post-clip (20 easy → 70 extreme)
- Destruction penalty: -30.0 unconditional (never anneals, never clips)
- Partial credit: 30.0 * (ratio^2) + (-5.0 * misses) on timeout/partial
- Step-level penalties anneal; terminal penalties never anneal

PHASE 3 GATES (per difficulty):
- Easy:     det>80%, gap<20%, ep_len>10
- Moderate: det>75%, gap<25%, ep_len>15
- Medium:   det>70%, gap<25%, ep_len>20
- Hard:     det>65%, gap<30%, ep_len>30
- Extreme:  det>55%, gap<35%, ep_len>40

HISTORY OF ALL ISSUES:
- IDLE WAIT abuse → REROUTE enforced from IDLE ✅
- Stationary agent → reward was always 0 ✅
- Reward always 0 → goal condition never triggered ✅
- Sparse reward / local optimum → curriculum + milestones ✅
- IN_TRANSIT WAIT=97.96% → TransitStallTracker ✅
- all_destroyed exploit (Phase 3) → terminal penalty floor + milestone gate ✅
- Hard=51% / Extreme=45% → reward scaling + asymmetric curriculum 🔧 ACTIVE FIX

NEXT TARGET:
- Hard delivery rate > 65%
- Extreme delivery rate > 55%
- Run diagnose_difficulty_failure() first to identify dominant termination
- Apply fixes in order: Fix 1 always, then targeted fix per diagnosis
```

---

*v5 — 5-difficulty curriculum | Hard/Extreme scaling fixes | Asymmetric exposure active*