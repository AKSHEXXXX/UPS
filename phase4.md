# ColdChain-Gym: Extreme Tier BC Alignment — What's Wrong & How To Fix It
**Session Type:** Targeted BC Alignment for Extreme Tier Deterministic Policy  
**Current State:** Hard=71% (healthy), Extreme det=0% stochastic=20%  
**Do Not:** Touch Hard tier training. Do not apply temperature scaling during training weights.  

---

## Your Current Plan And Exactly How Each Part Backfires

Read this section fully before touching any code. Every item in your current plan has a specific failure mode that will make things measurably worse, not better.

---

### ❌ Wrong Thing 1: BC on Harvested Stochastic Trajectories Without Quality Filtering

**What you're planning:**
Train BC on hard/extreme harvested states from best-episode trajectories using stochastic rollouts.

**How it backfires:**
Your stochastic policy delivers 20% on extreme. That means 80% of your harvested episodes are failures. "Best episode trajectories" from a 20% success rate pool means you are cloning the top 20% of bad outcomes — not successful behavior. The BC loss will minimize cross-entropy against whatever those failed episodes happened to do, which includes wrong routing decisions, ignored refrigeration degradation, and missed diversion calls. The margin loss will then sharpen the policy toward those wrong actions with high confidence. You will end up with a policy that is confidently wrong on extreme scenarios instead of uncertainly wrong. Deterministic extreme delivery will stay at 0% but with higher confidence in the wrong actions, making it harder to recover in the RL continuation phase.

**The specific failure signature you will see:**
After the BC pass, deterministic delivery on extreme stays at 0%. Stochastic delivery drops from 20% to ~10% because the BC pass also shifted probability mass away from the occasionally-correct stochastic actions. Hard tier may drop from 71% to 65% as a side effect.

---

### ❌ Wrong Thing 2: Action-Mask-Aware Logit Margin Loss Applied Episode-Wide

**What you're planning:**
Apply logit margin loss across full harvested trajectories to force top-1 separation on all states.

**How it backfires:**
Logit margin loss does not just nudge — it reshapes the entire logit landscape for every state it sees. Applied to full episode trajectories from a 20% success policy, most of the (state, action) pairs you are training on are states where the agent happened to take a mediocre action that didn't immediately cause failure. The margin loss will sharpen those mediocre actions to be the confident argmax. The policy will then execute those mediocre actions deterministically and consistently, eliminating the variance that currently allows stochastic sampling to occasionally find the right action. You are destroying your only working mechanism (stochastic variance) without replacing it with a better one.

**The specific failure signature you will see:**
Stochastic extreme drops significantly (20% → ~8%) because the variance that was enabling occasional correct behavior has been squeezed out. Deterministic extreme stays 0%. You now have a worse policy than before the BC pass in both modes.

---

### ❌ Wrong Thing 3: Temperature-Scaled Logits During BC Fine-Tune

**What you're planning:**
Apply temperature scaling to logits during the BC pass to increase top-1 minus top-2 margin on critical states.

**How it backfires in two separate ways:**

**Backfire A — It affects Hard tier even though you only intend it for Extreme.**
Temperature scaling during training modifies the weight updates globally. The policy does not know which tier it is being evaluated on during training. Your Hard tier policy learned flexible behavior at 71% — context-sensitive routing that sometimes picks the second-best action because the scenario calls for it. Temperature scaling during BC will sharpen this flexibility away. You will see Hard tier regress from 71% toward 60-65% because the policy now over-commits on scenarios where it should hedge.

**Backfire B — Temperature scaling amplifies whatever bias is already in the BC data.**
If your training data is biased toward wrong actions (which it is, given 80% failure rate in the harvest), temperature scaling makes the policy more confident in those wrong actions. Temperature scaling is a multiplier on existing logit separations — it does not create correct behavior, it amplifies whatever behavior already has the highest logit. On bad data, this means amplifying bad behavior.

**The specific failure signature you will see:**
Hard drops from 71% to ~62%. Extreme deterministic stays 0% but now with very high confidence on wrong actions. The policy is now harder to recover because the RL continuation needs to fight against a sharply-peaked wrong distribution instead of a flat uncertain one.

---

### ❌ Wrong Thing 4: 15-20k RL Steps After a Corrupted BC Pass

**What you're planning:**
Keep RL continuation short (15-20k steps) after the alignment pass, then re-evaluate.

**How it backfires:**
15-20k steps is calculated under the assumption that BC did something useful. If BC introduced bias (which it will, for the reasons above), 15-20k steps is not enough to correct the corruption. On extreme configs with max_steps=400 and n_steps=4096, 15k steps gives you approximately 37 full episodes. That is not enough for PPO to see enough successful extreme deliveries to unlearn the BC-induced bias. You will re-evaluate after 20k steps, see no improvement or regression, and conclude the approach failed — when actually the approach is sound but the execution corrupted the starting point.

**The specific failure signature you will see:**
After 20k RL steps, extreme deterministic is still 0%, Hard has regressed to ~63%, and you are now considering a full Phase 3 restart. The restart is not necessary — the architecture is fine — but the corrupted BC pass made it look necessary.

---

## The Root Cause Underneath All Four Mistakes

All four problems share a single root cause: **you are trying to apply BC alignment to a tier where you do not yet have good trajectories to clone from.**

BC is a supervised learning technique. It requires correct labeled examples. You have 20% stochastic success on extreme, which means your label quality is 20% at best and 0% at worst (because even the "successful" stochastic episodes may have succeeded via lucky sampling rather than correct decisions).

Before any BC pass, margin loss, or temperature scaling, you need to solve the data problem. Everything else is downstream of that.

---

## What You Should Do Instead

### Step 1: Generate Clean Extreme Trajectories First

Do not run any BC until this step produces trajectories with `DeliverySuccessGrader >= 0.40`.

**Method A — Greedy Rule-Based Solver (recommended, faster)**

Write a handcrafted dispatcher that generates extreme trajectories using explicit rules. This is not your trained policy — it is a simple rule-based system that you write in ~50 lines. It does not need to be good. It needs to be reliable enough to complete deliveries:

```python
def greedy_expert_action(env):
    """
    Rule-based dispatcher for trajectory harvesting only.
    Priority order — first matching rule fires:
    
    Rule 1: Vehicle has FAILED refrigeration + cargo onboard
            → DIVERT to nearest cold depot immediately, no exceptions
    
    Rule 2: Vehicle has DEGRADED refrigeration + detour_cost_to_depot < 8
            → DIVERT (cheap diversion, do it now before it fails)
    
    Rule 3: Any shipment has time_to_deadline < 10 steps + vehicle not at destination
            → EXPEDITE that vehicle if fuel allows, else REROUTE direct
    
    Rule 4: Vehicle is IDLE + undelivered shipments exist
            → REROUTE to destination of highest-priority undelivered shipment
    
    Rule 5: Vehicle IN_TRANSIT, no urgent conditions
            → WAIT (let it travel, do not thrash the route)
    """
    for vehicle in env.vehicles:
        if not getattr(vehicle, 'is_active', True):
            continue
        if vehicle.status.value == 3:  # BROKEN
            continue

        # Rule 1: Failed refrigeration is an emergency
        if vehicle.refrig_status.value == 2 and vehicle.shipments_onboard:
            return np.array([vehicle.id, 2, vehicle.nearest_cold_depot])

        # Rule 2: Degraded + cheap diversion
        if vehicle.refrig_status.value == 1 and vehicle.detour_cost < 8:
            return np.array([vehicle.id, 2, vehicle.nearest_cold_depot])

        # Rule 3: Deadline pressure
        for s_id in vehicle.shipments_onboard:
            s = env.shipments[s_id]
            if s.time_to_deadline < 10 and not s.is_delivered:
                if vehicle.fuel_level >= 0.1 and vehicle.status.value == 1:
                    return np.array([vehicle.id, 4, 0])  # EXPEDITE

        # Rule 4: Idle vehicle with work to do
        if vehicle.status.value == 0:
            undelivered = [s for s in env.shipments
                          if getattr(s, 'is_active', True)
                          and not s.is_delivered and not s.is_destroyed
                          and s.current_vehicle_id == -1]
            if undelivered:
                best = min(undelivered, key=lambda s: s.priority * -1)
                return np.array([vehicle.id, 1, best.destination_node])

    # Rule 5: No urgent action needed
    return np.array([env.config.n_vehicles, 0, 0])  # global no-op


def harvest_expert_trajectories(env_config, n_episodes=500, min_delivery_score=0.40):
    good_trajectories = []
    scores = []

    for seed in range(n_episodes):
        env = ColdChainEnv(config=env_config)
        obs, _ = env.reset(seed=7000 + seed)  # separate seed range
        trajectory = []
        done = False

        while not done:
            action = greedy_expert_action(env)
            obs, reward, terminated, truncated, info = env.step(action)
            trajectory.append((obs.copy(), action.copy(), reward, info))
            done = terminated or truncated

        score = DeliverySuccessGrader(trajectory).score()
        scores.append(score)

        if score >= min_delivery_score:
            good_trajectories.append(trajectory)
        env.close()

    print(f"Harvested {len(good_trajectories)}/{n_episodes} trajectories "
          f"above {min_delivery_score} threshold")
    print(f"Score distribution: "
          f"min={min(scores):.3f} mean={np.mean(scores):.3f} max={max(scores):.3f}")

    if len(good_trajectories) < 30:
        print("WARNING: Too few good trajectories. "
              "Lower min_delivery_score to 0.30 or run more episodes.")
    return good_trajectories
```

Run with extreme config. You need at least 50 good trajectories before proceeding.

**Method B — Rejection Sampling From Stochastic Policy**

If you prefer not to write the solver, run 1000 stochastic episodes and keep only the top 15% by CompositeGrader. At 20% stochastic success, 1000 episodes gives ~200 successes. Top 15% of those (~150 episodes) will have delivery ratios above 0.5. This is slower but requires no handcrafted logic.

```python
def rejection_sample_trajectories(model, env_config, n_episodes=1000, top_fraction=0.15):
    all_results = []

    for seed in range(n_episodes):
        env = ColdChainEnv(config=env_config)
        obs, _ = env.reset(seed=7000 + seed)
        trajectory = []
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=False)  # stochastic only
            obs, reward, terminated, truncated, info = env.step(action)
            trajectory.append((obs.copy(), action.copy(), reward, info))
            done = terminated or truncated

        score = CompositeGrader(trajectory).score()
        all_results.append((score, trajectory))
        env.close()

    all_results.sort(key=lambda x: x[0], reverse=True)
    cutoff = max(int(len(all_results) * top_fraction), 30)
    kept = all_results[:cutoff]

    print(f"Kept top {cutoff}/{n_episodes} trajectories")
    print(f"Score range of kept: "
          f"{kept[-1][0]:.3f} to {kept[0][0]:.3f}")
    return [t for _, t in kept]
```

---

### Step 2: Extract Only Critical Decision States

Do not run BC on full episode trajectories. Extract only the states where a difficult correct decision was made. This is the single change that prevents BC from corrupting Hard tier behavior.

```python
def extract_critical_states(trajectories):
    """
    Returns list of (obs, action, state_type) for states where
    a non-trivial correct decision was made.
    
    These are the only states where BC has leverage.
    BC on all states dilutes the signal and corrupts working behavior.
    """
    critical = []

    for trajectory in trajectories:
        for obs, action, reward, info in trajectory:
            vehicle_statuses = info.get("per_vehicle_status", {})
            shipment_statuses = info.get("per_shipment_status", {})
            action_type = action[1]
            vehicle_id = action[0]

            # Type 1: Correct diversion when refrigeration is not working
            v_data = vehicle_statuses.get(str(vehicle_id), {})
            refrig = v_data.get("refrig_status", 0)
            if refrig in [1, 2] and action_type == 2:  # DIVERT_COLD_DEPOT
                critical.append((obs.copy(), action.copy(), "diversion"))

            # Type 2: Triage — multiple critical shipments, chose highest priority routing
            critical_undelivered = sum(
                1 for s in shipment_statuses.values()
                if s.get("priority") == 2
                and not s.get("is_delivered", False)
                and not s.get("is_destroyed", False)
            )
            if critical_undelivered >= 2 and action_type == 1:  # REROUTE
                critical.append((obs.copy(), action.copy(), "triage"))

            # Type 3: Strategic abort — correctly gave up on a doomed shipment
            if action_type == 5:  # ABORT
                critical.append((obs.copy(), action.copy(), "abort"))

    type_counts = {}
    for _, _, t in critical:
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"Critical states extracted: {type_counts}")

    if type_counts.get("diversion", 0) < 30:
        print("WARNING: Fewer than 30 diversion states. "
              "BC may not fix refrigeration reaction. See Note A below.")
    return critical
```

**Note A:** If you get fewer than 30 diversion states from your harvested trajectories, the observation does not give the policy enough signal to distinguish degraded-refrigeration states. Before running BC, add the `thermal_risk_score` and `steps_until_cargo_breach` features to the observation (see Note B at end of this document).

---

### Step 3: BC With Margin Loss on Critical States Only

Now run BC. Three separate mini-passes, one per state type. Measure margin improvement after each.

```python
import torch
import torch.nn.functional as F

def margin_loss(logits, target_action, action_mask, margin=2.0):
    """
    Loss that pushes target_action logit to be at least `margin` above
    the next-best legal action. This is what forces top-1 separation.
    
    Do NOT use this on full episode trajectories.
    Only use on critical states where target_action is verified correct.
    """
    # Zero out illegal actions
    masked_logits = logits.clone()
    masked_logits[action_mask == 0] = -1e9

    target_logit = logits.gather(1, target_action.unsqueeze(1)).squeeze(1)

    # Best non-target logit among legal actions
    masked_logits_no_target = masked_logits.clone()
    masked_logits_no_target.scatter_(1, target_action.unsqueeze(1), -1e9)
    best_other_logit = masked_logits_no_target.max(dim=1).values

    # Margin loss: penalize when target is not `margin` above best other
    margin_violation = F.relu(best_other_logit - target_logit + margin)
    ce_loss = F.cross_entropy(logits, target_action)

    # Combined: cross-entropy for general fit + margin for top-1 separation
    return ce_loss + 0.5 * margin_violation.mean()


def measure_logit_margin(model, critical_states):
    """Check current top-1 minus top-2 margin before and after BC."""
    margins = []
    for obs, action, _ in critical_states[:100]:
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            logits = model.policy.get_distribution(obs_t).distribution.logits[0]
        sorted_logits = logits.sort(descending=True).values
        margins.append((sorted_logits[0] - sorted_logits[1]).item())
    return np.mean(margins)


def run_targeted_bc(model, critical_states, state_type_filter, 
                     n_epochs=3, lr=5e-5, batch_size=64):
    """
    BC pass on one critical state type only.
    Low learning rate — fine-tuning, not relearning.
    """
    states_of_type = [(o, a) for o, a, t in critical_states if t == state_type_filter]

    if len(states_of_type) < 20:
        print(f"Skipping BC for {state_type_filter}: "
              f"only {len(states_of_type)} examples (need >= 20)")
        return

    print(f"Running BC for {state_type_filter}: {len(states_of_type)} states")
    pre_margin = measure_logit_margin(model, 
                                       [(o, a, state_type_filter) 
                                        for o, a in states_of_type])
    print(f"  Pre-BC margin: {pre_margin:.3f}")

    optimizer = torch.optim.Adam(model.policy.parameters(), lr=lr)
    obs_t   = torch.FloatTensor(np.array([o for o, _ in states_of_type]))
    # Flatten MultiDiscrete action to single int for loss computation
    act_t   = torch.LongTensor(np.array([flatten_action(a) for _, a in states_of_type]))

    for epoch in range(n_epochs):
        perm = torch.randperm(len(states_of_type))
        total_loss = 0.0
        n_batches = 0

        for i in range(0, len(states_of_type), batch_size):
            idx = perm[i:i+batch_size]
            obs_b = obs_t[idx]
            act_b = act_t[idx]

            logits = model.policy.get_distribution(obs_b).distribution.logits
            # Use a placeholder full-legal mask if you don't store masks in trajectory
            # Ideally: store action_mask in info dict during harvesting
            action_mask = torch.ones_like(logits, dtype=torch.float)

            loss = margin_loss(logits, act_b, action_mask, margin=2.0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.3)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        print(f"  Epoch {epoch+1}/{n_epochs}: loss={total_loss/n_batches:.4f}")

    post_margin = measure_logit_margin(model,
                                        [(o, a, state_type_filter)
                                         for o, a in states_of_type])
    print(f"  Post-BC margin: {post_margin:.3f}  "
          f"(delta: {post_margin - pre_margin:+.3f})")

    if post_margin <= pre_margin + 0.2:
        print(f"  WARNING: Margin barely improved for {state_type_filter}. "
              f"Either data quality is low or observation lacks signal. "
              f"Do not proceed to RL continuation until this is resolved.")


def run_full_bc_alignment(model, critical_states):
    """Run all three BC passes in order. Measure Hard tier after each."""
    for state_type in ["diversion", "triage", "abort"]:
        run_targeted_bc(model, critical_states, state_type)
        # Quick Hard tier check after each pass
        hard_score = quick_eval(model, difficulty=3, n_episodes=10)
        print(f"  Hard tier quick-check after {state_type} BC: {hard_score:.3f}")
        if hard_score < 0.60:
            print(f"  STOP: Hard tier dropped below 0.60 after {state_type} BC. "
                  f"Restore from checkpoint and investigate data quality.")
            return False
    return True
```

The Hard tier check after each pass is non-negotiable. If Hard drops below 0.60 at any point, stop immediately and restore from the pre-BC checkpoint. The BC data has contaminated the policy.

---

### Step 4: Mandatory Go/No-Go Check Before RL Continuation

After all three BC passes complete without Hard regression, run this before touching RL:

```python
def bc_completion_check(model, extreme_config, hard_config):
    """
    Must pass all three conditions before RL continuation.
    If any fail, do not run RL — diagnose first.
    """
    print("=== BC COMPLETION CHECK ===")

    # Check 1: Deterministic extreme must be > 0%
    det_scores = []
    for seed in range(20):
        env = ColdChainEnv(config=extreme_config)
        obs, _ = env.reset(seed=9000 + seed)
        trajectory = []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            trajectory.append((obs, action, 0, info))
            done = terminated or truncated
        det_scores.append(DeliverySuccessGrader(trajectory).score())
        env.close()
    det_mean = np.mean(det_scores)
    check1 = det_mean > 0.05
    print(f"Check 1 — Det extreme delivery > 5%: {det_mean:.4f}  {'PASS' if check1 else 'FAIL'}")

    # Check 2: Hard tier must not have regressed below 0.65
    hard_scores = []
    for seed in range(15):
        env = ColdChainEnv(config=hard_config)
        obs, _ = env.reset(seed=9500 + seed)
        trajectory = []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            trajectory.append((obs, action, 0, info))
            done = terminated or truncated
        hard_scores.append(CompositeGrader(trajectory).score())
        env.close()
    hard_mean = np.mean(hard_scores)
    check2 = hard_mean >= 0.65
    print(f"Check 2 — Hard tier composite >= 0.65: {hard_mean:.4f}  {'PASS' if check2 else 'FAIL'}")

    # Check 3: Stochastic extreme must not have dropped more than 5 points
    sto_scores = []
    for seed in range(20):
        env = ColdChainEnv(config=extreme_config)
        obs, _ = env.reset(seed=9000 + seed)
        trajectory = []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, _, terminated, truncated, info = env.step(action)
            trajectory.append((obs, action, 0, info))
            done = terminated or truncated
        sto_scores.append(DeliverySuccessGrader(trajectory).score())
        env.close()
    sto_mean = np.mean(sto_scores)
    check3 = sto_mean >= 0.15  # was 0.20, allow small drop
    print(f"Check 3 — Stochastic extreme delivery >= 15%: {sto_mean:.4f}  {'PASS' if check3 else 'FAIL'}")

    all_pass = check1 and check2 and check3
    print(f"\nOverall BC check: {'PROCEED TO RL' if all_pass else 'DO NOT PROCEED — RESTORE CHECKPOINT'}")
    return all_pass
```

If any check fails, restore from the pre-BC checkpoint and identify which BC pass caused the regression. Do not run RL continuation on a corrupted policy.

---

### Step 5: RL Continuation Length — Conditional on BC Check

```
BC check all pass?
    ↓
YES → det_extreme was > 10%?
         YES → 15-20k RL steps (consolidate BC gains)
         NO  → 15k RL steps + add refrigeration reaction reward (Note B)
    ↓
NO  → Restore checkpoint → diagnose which BC pass failed → fix data → retry Step 1
```

RL continuation config (do not change from Phase 3 settings):
```python
rl_continuation_config = {
    "total_timesteps": 20_000,
    "ent_coef":        0.01,    # low — consolidating, not exploring
    "learning_rate":   1e-5,    # very low — fine-tuning
    "clip_range":      0.10,    # tight — preserve BC gains
    "n_steps":         4096,
    "batch_size":      256,
}
```

---

### Step 6: Temperature Scaling at Inference Only — Not During Training

Apply this only during evaluation of Extreme tier. Do not touch training weights.

```python
class TemperatureScaledPredictor:
    """
    Wraps a trained model and applies temperature scaling at inference.
    Temperature < 1.0 sharpens the distribution (use for Extreme eval).
    Does NOT modify model weights. Safe for Hard tier — use temperature=1.0 there.
    """
    def __init__(self, model, temperature=0.7):
        self.model = model
        self.temperature = temperature

    def predict(self, obs, deterministic=True):
        import torch
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            dist = self.model.policy.get_distribution(obs_t)
            logits = dist.distribution.logits[0]

        scaled = logits / self.temperature
        scaled[~self._get_mask(obs)] = -1e9  # reapply action mask

        if deterministic:
            action_flat = scaled.argmax().item()
        else:
            action_flat = torch.distributions.Categorical(logits=scaled).sample().item()

        return unflatten_action(action_flat), None

    def _get_mask(self, obs):
        # Reconstruct env state from obs to get mask — or cache from last env.step()
        # Implementation depends on your env wrapper
        pass

# Usage:
sharp_predictor = TemperatureScaledPredictor(model, temperature=0.7)

# Extreme evaluation only:
result = ExtremeGrader(sharp_predictor).evaluate(n_episodes=30)

# Hard evaluation — standard temperature:
standard_predictor = TemperatureScaledPredictor(model, temperature=1.0)
result = HardGrader(standard_predictor).evaluate(n_episodes=30)
```

Tune temperature between 0.5 and 0.9 on 10 evaluation episodes before the official grader run. `temperature=0.7` is a good starting point. If extreme deterministic delivery does not improve with temperature scaling, the issue is not distribution flatness — it is a wrong-mode problem (the most probable action is genuinely wrong) and you need Note B.

---

## Execution Order — Do Not Skip Steps

```
Step 1: Harvest clean extreme trajectories (greedy solver or rejection sampling)
        → Verify: at least 50 trajectories with DeliverySuccessGrader >= 0.40
        → If < 50: lower threshold to 0.30 and try again OR see Note B

Step 2: Extract critical states from harvested trajectories
        → Verify: at least 30 diversion states, 20 triage states
        → If not: see Note A

Step 3A: Checkpoint current model BEFORE any BC (name it: bc_pre_alignment.zip)
Step 3B: Run BC for "diversion" states → check Hard tier → stop if Hard < 0.65
Step 3C: Run BC for "triage" states → check Hard tier → stop if Hard < 0.65
Step 3D: Run BC for "abort" states → check Hard tier → stop if Hard < 0.65

Step 4: Run bc_completion_check() → all three conditions must pass
        → If any fail: restore bc_pre_alignment.zip, diagnose, fix data

Step 5: RL continuation (15-20k steps) with tight config above
        → DeterministicDeliveryCallback active throughout

Step 6: Temperature-scaled evaluation (temperature=0.7) on Extreme tier
        → Standard evaluation on Hard tier

Step 7: Run official tiered graders → report
```

---

## What To Report After Each Step

```
STEP 1 COMPLETE:
  Trajectories harvested: [N] total, [N] above threshold
  Score range: min=[X] mean=[X] max=[X]
  Method used: greedy_solver / rejection_sampling

STEP 2 COMPLETE:
  Diversion states: [N]
  Triage states: [N]
  Abort states: [N]

STEP 3 COMPLETE (after all three BC passes):
  Diversion BC: pre_margin=[X] post_margin=[X] Hard_after=[X]
  Triage BC: pre_margin=[X] post_margin=[X] Hard_after=[X]
  Abort BC: pre_margin=[X] post_margin=[X] Hard_after=[X]
  Any Hard regression? YES (stopped + restored) / NO

STEP 4 — BC COMPLETION CHECK:
  Det extreme delivery: [X]  PASS/FAIL
  Hard composite: [X]  PASS/FAIL
  Stochastic extreme: [X]  PASS/FAIL
  Overall: PROCEED / RESTORE

STEP 5 COMPLETE (after RL continuation):
  Det extreme at end: [X]
  Hard at end: [X]

STEP 6 — TEMPERATURE TUNING:
  Best temperature found: [X]
  Det extreme with temperature=[X]: [X]

STEP 7 — OFFICIAL GRADER RESULTS:
  Easy: [X]
  Moderate: [X]
  Hard: [X]
  Extreme: [X]
```

---

## Note A: If You Cannot Get 30 Diversion States

The observation does not expose refrigeration degradation clearly enough for the policy to distinguish those states. Add these two features to the vehicle observation row before running BC:

```python
# In _get_obs(), for each vehicle:

# Feature 1: thermal_risk_score — single number summarizing danger level
thermal_risk = {0: 0.0, 1: 0.5, 2: 1.0}[vehicle.refrig_status.value]

# Feature 2: steps_until_cargo_breach — explicit countdown
steps_until_breach = 999.0
for s_id in vehicle.shipments_onboard:
    s = shipments[s_id]
    if vehicle.refrig_status.value > 0:  # not working perfectly
        drift_per_step = abs(outdoor_temp - s.cargo_temp) * k * step_duration
        if drift_per_step > 0 and s.cargo_temp < s.temp_upper_bound:
            headroom = s.temp_upper_bound - s.cargo_temp
            steps = headroom / drift_per_step
            steps_until_breach = min(steps_until_breach, steps)
# Normalize to [0, 1]: 0 = breach now, 1 = breach in 20+ steps
normalized_breach = min(steps_until_breach / 20.0, 1.0)
```

After adding these features, re-harvest trajectories. The greedy solver will now generate diversion states that the BC training can distinguish.

---

## Note B: If Det Extreme Is Still 0% After Full BC + RL

The policy is not failing due to distribution flatness — it is failing because the reward signal for refrigeration reaction is too weak for the gradient to have ever learned it. Add this to `reward.py` before any further training:

```python
def refrigeration_reaction_reward(vehicle, shipments, action_type):
    if vehicle.refrig_status.value == 0:  # WORKING — no signal needed
        return 0.0
    has_active_cargo = any(
        not shipments[s_id].is_destroyed and not shipments[s_id].is_delivered
        for s_id in vehicle.shipments_onboard
    )
    if not has_active_cargo:
        return 0.0
    if action_type == 2:   # DIVERT_COLD_DEPOT — correct
        urgency = 1.0 if vehicle.refrig_status.value == 2 else 0.5
        return +0.30 * urgency
    elif action_type == 1:  # REROUTE despite degraded/failed — wrong
        urgency = 1.0 if vehicle.refrig_status.value == 2 else 0.3
        return -0.15 * urgency
    return 0.0
```

If this reward is needed, restart RL continuation from the post-BC checkpoint with this reward added. Do not restart from Phase 2.