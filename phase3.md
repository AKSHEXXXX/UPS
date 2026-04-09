# ColdChain-Gym: Hard/Extreme Tier Failure Analysis & Solutions
**Date:** Post Phase-2 Redesign  
**Status:** Architecture complete. Policy quality pending. Phase-3 tuning required.  
**Author:** Engineering Review  

---

## 1. Honest Situation Assessment

The redesign succeeded at everything it was supposed to do. The pipeline is clean, the curriculum is real, the observation contract is fixed, the tests pass, and checkpoints resume correctly. None of that is the problem anymore.

The problem is one specific and well-understood failure mode: **deterministic delivery is 0% while stochastic delivery remains high.** This gap exists in both Phase 1 and Phase 2 validation. It has nothing to do with architecture. It is a policy quality problem with a known cause and a known fix path.

This document identifies the root causes precisely and gives the Phase-3 tuning instructions to close the gap.

---

## 2. Root Cause Analysis

### 2.1 The Stochastic/Deterministic Gap — What It Actually Means

Deterministic evaluation means `model.predict(obs, deterministic=True)` — argmax over the action distribution. Stochastic means sampling from it.

**Stochastic high + Deterministic zero means exactly one thing:** the policy learned a *multimodal distribution* where the correct action has real probability mass but is not the mode. Argmax consistently picks a different action — one that is catastrophic in practice.

This is not "the model hasn't learned." The model learned something. It just learned to spread probability across several actions rather than committing to the right one. The cause is almost always **entropy regularization keeping the distribution flat beyond the point where it should commit.**

If you log the top-3 action probabilities during a deterministic failure episode, you will see something like:

```
Step 12:  REROUTE(node_7)=0.31   REROUTE(node_3)=0.28   DIVERT_DEPOT=0.22
Step 13:  REROUTE(node_7)=0.33   REROUTE(node_3)=0.26   DIVERT_DEPOT=0.21
```

The correct action (DIVERT_DEPOT when refrigeration just degraded) has 22% probability — enough to fire often in stochastic rollouts — but argmax always picks REROUTE. Every deterministic episode ignores refrigeration degradation and destroys the cargo.

### 2.2 The "all_destroyed" Termination Pattern

`all_destroyed` as the dominant failure mode in deterministic episodes tells you the specific mechanism:

1. Vehicle starts journey. Refrigeration degrades (stochastic event mid-route).
2. Stochastic policy: sometimes samples DIVERT_COLD_DEPOT in time. Works.
3. Deterministic policy: always picks REROUTE (highest logit). Never diverts. Cargo temp climbs. Destruction.

The deterministic policy has **no reactive behavior to environmental state changes.** It commits to a route and executes it regardless of what happens to the refrigeration unit. The stochastic policy accidentally does the right thing by sampling. This means the reward signal for reacting to refrigeration degradation is too weak to push DIVERT_COLD_DEPOT to be the argmax.

### 2.3 The Curriculum Transition Is Too Abrupt

Phase 2 trained primarily on easy/moderate scenarios. Phase 3 immediately biases toward hard/extreme. The deterministic policy learned habits that work on easy scenarios (straight routing, minimal diversion) but are lethal on hard/extreme where:

- Deadlines give almost no slack
- Refrigeration fails much faster  
- Weather multiplies travel time by 2.5x

The deterministic policy's "default behavior" is optimized for the easy case. Under hard conditions it executes that same default behavior and fails catastrophically. The stochastic policy's variance covers for this sometimes.

### 2.4 Difficulty Reward Scaling Miscalibration

Expanding difficulty scaling from 3 to 5 levels changes the reward magnitudes the value function has seen. If Phase 3 samples hard/extreme scenarios (difficulty 4–5) but the value function was trained on difficulty 1–3, the value estimates for hard scenarios are wrong. PPO uses these value estimates to compute advantages. Wrong advantages → wrong gradient direction → the policy updates away from the correct action even when it was working.

This explains why Phase 2 validation at difficulty 3 also showed 0% deterministic delivery — the value function is miscalibrated for scenarios harder than what Phase 1 primarily trained on.

### 2.5 The is_active Flag May Be Confusing Action Selection

The new is_active flag marks extra vehicles/shipments inactive in the fixed tensor. If the action mask does not perfectly zero out actions targeting inactive entities, the deterministic policy may be consistently choosing to interact with inactive slots — particularly if those slots have slightly higher logits due to their zero-padded observation features being a distribution the network learned to respond to.

This is a lower-probability cause but worth verifying because it would explain perfectly reproducible 0% delivery across all seeds.

---

## 3. Fix Arsenal — Ordered by Priority

### Fix 1 (CRITICAL — Do This First): Entropy Annealing in Phase 3

The policy needs to commit. Stop rewarding entropy past the point of useful exploration.

```python
# In your PPO Phase 3 config:

# Start Phase 3 with reduced entropy — force commitment
ent_coef_start = 0.02    # was 0.05 during Phase 2 — already lower
ent_coef_end   = 0.005   # anneal toward near-zero by end of Phase 3

# Implement linear annealing via callback:
class EntropyAnnealCallback(BaseCallback):
    def __init__(self, ent_start, ent_end, total_steps):
        super().__init__()
        self.ent_start = ent_start
        self.ent_end = ent_end
        self.total_steps = total_steps

    def _on_step(self):
        fraction = self.n_calls / self.total_steps
        new_ent = self.ent_start + fraction * (self.ent_end - self.ent_start)
        self.model.ent_coef = max(new_ent, self.ent_end)
        return True
```

This forces the policy to sharpen its action distribution over Phase 3. By the end, argmax and sampling should converge. If they don't converge by 50% of Phase 3 training, the reward signal for the right action is too weak — see Fix 3.

**Expected effect:** Deterministic delivery should become non-zero within the first 20% of Phase 3 training.

---

### Fix 2 (CRITICAL): Add Deterministic Evaluation Callback During Training

Right now you only discover the deterministic gap at phase gates. That means you run thousands of steps without knowing the deterministic policy is failing. Add this callback so you see the gap in real time:

```python
class DeterministicDeliveryCallback(BaseCallback):
    """
    Runs N deterministic episodes every eval_freq steps.
    Logs deterministic delivery ratio so you can track the
    stochastic/deterministic gap closing in real time.
    """
    def __init__(self, eval_envs_by_difficulty, eval_freq=2000, n_eval=10):
        super().__init__()
        self.eval_envs = eval_envs_by_difficulty   # dict: {difficulty: env_config}
        self.eval_freq = eval_freq
        self.n_eval = n_eval

    def _on_step(self):
        if self.n_calls % self.eval_freq != 0:
            return True

        for difficulty, config in self.eval_envs.items():
            det_deliveries = []
            sto_deliveries = []

            for seed in range(self.n_eval):
                env = ColdChainEnv(config=config)

                # Deterministic
                obs, _ = env.reset(seed=8000 + seed)
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                summary = info.get("episode_summary", {})
                det_deliveries.append(
                    summary.get("delivered_count", 0) / max(summary.get("total_shipments", 1), 1)
                )

                # Stochastic
                obs, _ = env.reset(seed=8000 + seed)
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=False)
                    obs, _, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                summary = info.get("episode_summary", {})
                sto_deliveries.append(
                    summary.get("delivered_count", 0) / max(summary.get("total_shipments", 1), 1)
                )

                env.close()

            det_mean = np.mean(det_deliveries)
            sto_mean = np.mean(sto_deliveries)
            gap = sto_mean - det_mean

            print(f"  Step {self.n_calls} | diff={difficulty} | "
                  f"det={det_mean:.3f} sto={sto_mean:.3f} gap={gap:.3f}")

            # Alert if gap is still large after 30% of Phase 3
            if self.n_calls > 0.3 * self.locals.get("total_timesteps", 1e6):
                if gap > 0.30:
                    print(f"  WARNING: Stochastic/deterministic gap {gap:.3f} "
                          f"still large at {self.n_calls} steps. "
                          f"Entropy annealing may need to be more aggressive.")
        return True
```

**Why this is critical:** Without this, you run Phase 3 to completion and only find out at the gate that deterministic still fails. With this callback, you know within the first 4000 steps whether Phase 3 is closing the gap.

---

### Fix 3 (HIGH): Strengthen the Refrigeration Reaction Signal

The `all_destroyed` failure mode means the agent isn't reacting to refrigeration degradation. The reward signal for reacting is too weak relative to the reward for continuing the route. Fix both the observation salience and the reward:

**Observation change — make degradation impossible to miss:**
```python
# In _get_obs(), for each vehicle row, add a derived feature:
# "thermal_risk_score" — combines refrig status + cargo temp drift rate
# This puts the danger signal in ONE place the network can easily attend to

thermal_risk = 0.0
if vehicle.refrig_status == RefrigStatus.DEGRADED:
    thermal_risk = 0.5
elif vehicle.refrig_status == RefrigStatus.FAILED:
    thermal_risk = 1.0

# Also add: steps_until_cargo_destroyed (estimated)
# = (temp_upper_bound - cargo_temp) / drift_rate_per_step
# This gives the agent an explicit countdown
for s_id in vehicle.shipments_onboard:
    s = shipments[s_id]
    if vehicle.refrig_status != RefrigStatus.WORKING:
        drift = (outdoor_temp - s.cargo_temp) * k * step_duration
        if drift > 0:  # cargo is warming
            headroom = s.temp_upper_bound - s.cargo_temp
            steps_until_breach = max(headroom / drift, 0)
            # Normalize to [0, 1]: 0 = breach imminent, 1 = safe for many steps
            normalized = min(steps_until_breach / 20.0, 1.0)
            # Add to shipment observation row
```

**Reward change — explicit diversion reward when refrigeration is degraded:**
```python
def refrig_reaction_reward(vehicle, shipments, action_type, config):
    """
    Reward the agent specifically for diverting when refrigeration is degraded.
    This creates a strong signal that DIVERT_COLD_DEPOT is right when refrig != WORKING.
    """
    if vehicle.refrig_status == RefrigStatus.WORKING:
        return 0.0  # no reward needed — refrigeration is fine

    has_active_cargo = any(
        not shipments[s_id].is_destroyed and not shipments[s_id].is_delivered
        for s_id in vehicle.shipments_onboard
    )
    if not has_active_cargo:
        return 0.0

    if action_type == ActionType.DIVERT_COLD_DEPOT:
        # Agent is doing the right thing — reward proportional to urgency
        urgency = 1.0 if vehicle.refrig_status == RefrigStatus.FAILED else 0.5
        return +0.30 * urgency
    elif action_type == ActionType.REROUTE:
        # Agent is continuing despite degraded/failed refrigeration — penalize
        urgency = 1.0 if vehicle.refrig_status == RefrigStatus.FAILED else 0.3
        return -0.15 * urgency
    return 0.0
```

The asymmetry (+0.30 for correct diversion, -0.15 for wrong continuation) creates a strong push toward diversion. Currently this signal is implicit in the eventual cargo destruction penalty — by then it's too late for the policy gradient to attribute cause correctly.

---

### Fix 4 (HIGH): Warm Start Phase 3 With Best Stochastic Trajectories

Phase 3 starts from Phase 2's checkpoint. That checkpoint's deterministic policy delivers 0%. Instead of hoping Phase 3 RL will fix this from scratch, give it a head start by doing a short behavioral cloning pass on Phase 2's best stochastic rollouts first.

```python
def collect_best_stochastic_rollouts(model, config, n_episodes=200, top_fraction=0.3):
    """
    Run stochastic episodes, keep the top 30% by delivery ratio.
    Returns list of (obs, action) pairs for BC.
    """
    rollouts = []
    for seed in range(n_episodes):
        env = ColdChainEnv(config=config)
        obs, _ = env.reset(seed=seed)
        episode = []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=False)  # stochastic
            next_obs, _, terminated, truncated, info = env.step(action)
            episode.append((obs.copy(), action.copy()))
            obs = next_obs
            done = terminated or truncated
        summary = info.get("episode_summary", {})
        delivery_ratio = summary.get("delivered_count", 0) / max(summary.get("total_shipments", 1), 1)
        rollouts.append((delivery_ratio, episode))
        env.close()

    # Keep only top-performing rollouts
    rollouts.sort(key=lambda x: x[0], reverse=True)
    cutoff = int(len(rollouts) * top_fraction)
    best_rollouts = rollouts[:cutoff]
    print(f"BC warmup: kept {cutoff}/{n_episodes} rollouts "
          f"(min delivery ratio: {best_rollouts[-1][0]:.3f})")

    return [(obs, action) for _, episode in best_rollouts for obs, action in episode]


def bc_warmup(model, bc_data, n_epochs=3, lr=1e-4):
    """
    Short supervised pass to push the policy toward high-delivery actions.
    Modifies model in-place. Run before Phase 3 RL training.
    """
    import torch
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=lr)

    obs_batch = torch.FloatTensor(np.array([d[0] for d in bc_data]))
    act_batch = torch.LongTensor(np.array([d[1] for d in bc_data]))

    dataset_size = len(bc_data)
    batch_size = 256

    for epoch in range(n_epochs):
        perm = torch.randperm(dataset_size)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, dataset_size, batch_size):
            idx = perm[i:i+batch_size]
            obs_b = obs_batch[idx]
            act_b = act_batch[idx]

            log_probs = model.policy.get_distribution(obs_b).log_prob(act_b)
            loss = -log_probs.mean()   # maximize log probability of good actions

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        print(f"  BC epoch {epoch+1}/{n_epochs}: mean loss={total_loss/n_batches:.4f}")

# Usage before Phase 3 training:
hard_config = ColdChainConfig(n_vehicles=3, n_shipments=5, ...)
bc_data = collect_best_stochastic_rollouts(phase2_model, hard_config, n_episodes=300)
bc_warmup(phase2_model, bc_data, n_epochs=5)
# Now start Phase 3 RL from this warmed model
```

This should immediately raise deterministic delivery above zero because the policy has been pushed toward the action distribution of successful stochastic rollouts before RL even starts.

---

### Fix 5 (MEDIUM): Phase 3 Curriculum Should Ramp, Not Jump

Phase 3 "heavily biases sampling toward hard/extreme" — but the model starts Phase 3 with a policy that only reliably works on easy/moderate. Throwing it straight into hard/extreme is like removing training wheels and putting the rider on a mountain. The policy doesn't have the gradient signal to adapt fast enough.

Replace the heavy bias with a **linear ramp over the first 30% of Phase 3:**

```python
class CurriculumRampScheduler:
    """
    Phase 3 curriculum: start at moderate-heavy mix, ramp to extreme-heavy.
    
    At Phase 3 start (step 0):
      easy=0.05, moderate=0.20, medium=0.30, hard=0.30, extreme=0.15
    At Phase 3 midpoint (step total/2):
      easy=0.00, moderate=0.05, medium=0.15, hard=0.40, extreme=0.40
    At Phase 3 end:
      easy=0.00, moderate=0.00, medium=0.10, hard=0.35, extreme=0.55
    """
    def __init__(self, total_steps):
        self.total_steps = total_steps
        # Each entry: (easy, moderate, medium, hard, extreme)
        self.start_weights = (0.05, 0.20, 0.30, 0.30, 0.15)
        self.end_weights   = (0.00, 0.00, 0.10, 0.35, 0.55)

    def get_weights(self, current_step):
        t = min(current_step / self.total_steps, 1.0)
        weights = tuple(
            s + t * (e - s)
            for s, e in zip(self.start_weights, self.end_weights)
        )
        total = sum(weights)
        return tuple(w / total for w in weights)

    def sample_difficulty(self, current_step, rng=None):
        weights = self.get_weights(current_step)
        rng = rng or np.random.default_rng()
        difficulties = [1, 2, 3, 4, 5]
        return rng.choice(difficulties, p=weights)
```

The first 30% of Phase 3 is still mostly medium/hard (60%), giving the policy gradient time to adapt before you start hammering it with extreme. This is not softening the curriculum — it's pacing it correctly.

---

### Fix 6 (MEDIUM): Verify is_active Masking Is Correct

Before running Phase 3, run this diagnostic to confirm inactive entities are not being targeted:

```python
def audit_inactive_action_targeting(model, n_episodes=50):
    """
    Checks whether the deterministic policy is targeting inactive vehicles/shipments.
    If inactive_targeting_rate > 5%, the is_active masking has a bug.
    """
    config = ColdChainConfig(n_vehicles=5, n_shipments=8)  # full tensor size
    # But only activate 2 vehicles and 3 shipments
    small_config = ColdChainConfig(n_vehicles=2, n_shipments=3)
    
    inactive_actions = 0
    total_actions = 0
    
    for seed in range(n_episodes):
        env = ColdChainEnv(config=config)  # full tensor
        obs, _ = env.reset(seed=seed)
        # Manually mark vehicles 2-4 and shipments 3-7 as inactive
        for i in range(2, 5):
            env.vehicles[i].is_active = False
        for i in range(3, 8):
            env.shipments[i].is_active = False
        
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            vehicle_idx = action[0]
            
            # Check if targeting inactive vehicle
            if vehicle_idx < len(env.vehicles) and not env.vehicles[vehicle_idx].is_active:
                inactive_actions += 1
            
            total_actions += 1
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        env.close()
    
    rate = inactive_actions / max(total_actions, 1)
    print(f"Inactive entity targeting rate: {rate:.4f} ({inactive_actions}/{total_actions})")
    if rate > 0.05:
        print("WARNING: Policy is targeting inactive entities. "
              "Action mask for is_active=False entities needs to be verified.")
    else:
        print("OK: Inactive targeting rate is acceptable.")
    return rate
```

If the rate is above 5%, fix the action mask to zero out all actions targeting `is_active=False` entities before running Phase 3. This is a silent killer — if deterministic policy always picks an inactive vehicle, delivery is structurally impossible.

---

### Fix 7 (MEDIUM): Value Function Recalibration at Phase 3 Start

The value function was calibrated on difficulty 1–3 rewards. Phase 3 introduces difficulty 4–5 which has different reward magnitudes. Mismatched value estimates produce wrong advantages, which corrupt the gradient.

Apply value function pretraining before Phase 3 policy learning:

```python
# In PPO Phase 3 config — increase value function learning rate temporarily:

# Option A: Higher vf_coef at start of Phase 3, lower after warmup
vf_coef_schedule = {
    0:       0.7,   # high at start — recalibrate value function quickly
    100000:  0.5,   # back to normal after 100k steps
}

# Option B: Run value function only updates for first 5k steps
# by setting policy_gradient_steps=0 and only doing value updates
# (depends on your SB3 version / custom PPO)

# Option C: Normalize rewards per difficulty level so value function
# sees consistent scale regardless of difficulty
def normalize_reward_by_difficulty(reward, difficulty):
    # Difficulty 1-5 have different reward magnitudes
    # Normalize to roughly same scale
    scale_factors = {1: 1.0, 2: 0.9, 3: 0.8, 4: 0.75, 5: 0.7}
    return reward * scale_factors.get(difficulty, 1.0)
```

Option C is the easiest to implement and doesn't require PPO internals access.

---

## 4. Phase 3 Configuration — Recommended Settings

Apply all of the following together. These are not independent — they are designed as a coordinated set.

```python
phase3_ppo_config = {
    # Exploration: annealing from moderate to minimal
    "ent_coef":          0.02,         # start lower than Phase 2
    # (entropy annealing callback reduces this to 0.005 by end)

    # Rollout: longer for sparse delivery signal
    "n_steps":           4096,

    # Optimization: stable gradients
    "batch_size":        512,
    "n_epochs":          15,

    # Value function: higher coef to recalibrate quickly
    "vf_coef":           0.6,          # start high, reduce after 100k steps

    # Long horizon planning
    "gamma":             0.997,        # slightly higher than Phase 2
    "gae_lambda":        0.97,

    # Learning rate: lower than Phase 2 — fine-tuning not relearning
    "learning_rate":     3e-5,         # was likely 1e-4 in Phase 2

    # Clip: tighter — preserve Phase 2 knowledge, don't overwrite
    "clip_range":        0.15,         # was likely 0.2

    # Total steps: enough for curriculum ramp to take effect
    "total_timesteps":   2_000_000,
}
```

**Callbacks to attach:**
1. `EntropyAnnealCallback(ent_start=0.02, ent_end=0.005, total_steps=2_000_000)`
2. `DeterministicDeliveryCallback(eval_envs, eval_freq=2000)`
3. `DeliveryRatioCallback` (from previous session)
4. `CurriculumRampScheduler` integrated into scenario sampler

---

## 5. Diagnostic Sequence Before Starting Phase 3

Run these in order. Do not start Phase 3 training until all four return clean results.

```
1. audit_inactive_action_targeting(phase2_model)
   → Target: inactive targeting rate < 5%
   → If fails: fix action mask for is_active entities

2. Run DeterministicDeliveryCallback manually for 1 episode at each difficulty
   → Confirms the gap exists and at which difficulty it breaks
   → Expected: difficulty 1-2 might work, 3-5 will be 0%

3. Run bc_warmup with best stochastic rollouts from Phase 2
   → After warmup: rerun deterministic eval at difficulty 3
   → Expected: delivery > 0% after BC warmup

4. Confirm reward normalization across difficulties
   → Print mean reward per difficulty level across 20 stochastic episodes each
   → If difficulty-5 mean reward is 3x difficulty-1, apply normalization
```

---

## 6. What Success Looks Like at Phase 3 Completion

| Metric | Current | Target |
|---|---|---|
| Easy tier score | ~85% | >85% (maintain) |
| Moderate tier score | ~85% | >80% (slight regression acceptable) |
| Hard tier score | ~53% | >60% |
| Extreme tier score | ~41% | >45% |
| Deterministic delivery at difficulty 3 | 0% | >40% |
| Deterministic delivery at difficulty 5 | 0% | >20% |
| Stochastic/deterministic gap at difficulty 3 | >50% | <20% |
| Organ survival on extreme tier | unknown | >40% |

These targets are achievable with Phase 3 alone. They do not require further architectural changes.

If after full Phase 3 training deterministic delivery at difficulty 3 is still below 20%, the root cause is Fix 3 (refrigeration reaction signal) and it needs to be applied mid-Phase-3. Do not wait until Phase 3 completes to check.

---

## 7. What NOT To Do

These are tempting interventions that will make things worse.

**Do not increase adversarial severity in Phase 3.** The adversarial variants are already producing all_destroyed failures in Phase 2. Making them more severe will only deepen the gradient confusion. The adversarial seeds are working — the policy just hasn't learned to handle them yet.

**Do not restart from Phase 1.** Phase 2's stochastic policy has real knowledge. The BC warmup preserves it. Starting over discards two phases of learning to fix a tuning problem.

**Do not reduce max_steps to force faster episodes.** Shorter episodes will reduce the already-sparse delivery signal further. The agent needs to experience full delivery sequences to learn them.

**Do not remove entropy regularization entirely.** Setting `ent_coef=0` will cause PPO to collapse to a deterministic policy too fast, before the right action is the mode. Use the annealing schedule — taper to 0.005, not to zero.

**Do not add more observation features until inactive masking is verified.** If the is_active flag masking is broken (Fix 6), adding more features just gives the broken policy more to overfit on.

---

## 8. Decision

The architecture is sound. The curriculum, observation contract, reward shaping, and test suite are all in the right place. The remaining problem — deterministic policy delivering 0% under difficulty 3–5 — is a policy quality issue caused by insufficient entropy annealing, a missing refrigeration reaction signal, and an abrupt curriculum ramp. These are tuning problems, not structural ones.

**Proceed with Phase 3 training** using the configuration in Section 4, the BC warmup in Fix 4, and the entropy annealing in Fix 1. Run the diagnostic sequence in Section 5 before starting. Monitor the stochastic/deterministic gap with the callback in Fix 2 from the first step.

Do not do a further structural refactor. The next decision point is after Phase 3 completes: if hard tier is above 60% and extreme above 45%, the environment ships. If not, apply Fix 3 (refrigeration reaction reward) as a targeted mid-Phase-3 intervention and retrain from the Phase 3 midpoint checkpoint — not from scratch.