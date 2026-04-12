# Fix: inference.py — OpenEnv Validator [END] Format & Score Clamping
**File to edit:** `inference.py` (root of the UPS repo, branch: Master)  
**Problem:** OpenEnv Phase 2 Task Validation fails with "Not enough tasks with graders"  
**Root cause:** Two bugs in the `[END]` output line  

---

## Context — Why It's Failing

The OpenEnv validator parses the stdout of `inference.py` using a strict regex.
It expects the `[END]` line to look **exactly** like this:

```
[END] success=true steps=20 score=0.850 rewards=0.00,0.10,0.90
```

Your current `[END]` line outputs extra fields that break the parser:

```
[END] success=true steps=20 tasks=4 termination_reason=all_tasks_completed score=0.85 delivery_score=0.90 thermal_score=0.82 efficiency_score=0.78 rewards=...
```

The validator cannot find a valid `score=` in the expected position because extra fields precede it. It therefore reads zero valid grader outputs and throws "not enough tasks with graders" — even though your graders work perfectly.

A second bug: `score` can be exactly `0.0` or `1.0`. The validator rejects scores at the exact boundaries. Score must be clamped to `(1e-6, 1-1e-6)` — confirmed by the Discord thread and validated against the working reference repo.

---

## Bug 1 — Wrong `[END]` format

### Location
`inference.py`, inside the `finally:` block of `main()`, approximately lines 486–498.

### Current broken code
```python
        reward_text = ",".join(_fmt_float(reward) for reward in rewards)
        print(
            "[END] "
            f"success={_bool_text(success)} "
            f"steps={steps} "
            f"tasks={len(task_sequence)} "
            f"termination_reason={termination_reason} "
            f"score={_fmt_float(end_score)} "
            f"delivery_score={_fmt_float(float(grader_scores.get('delivery', 0.0)))} "
            f"thermal_score={_fmt_float(float(grader_scores.get('thermal', 0.0)))} "
            f"efficiency_score={_fmt_float(float(grader_scores.get('efficiency', 0.0)))} "
            f"rewards={reward_text}",
            flush=True,
        )
```

### Fixed code — replace the entire block above with this
```python
        reward_text = ",".join(_fmt_float(reward) for reward in rewards)
        score_clamped = max(1e-6, min(end_score, 1 - 1e-6))
        print(
            f"[END] success={_bool_text(success)} "
            f"steps={steps} "
            f"score={score_clamped:.3f} "
            f"rewards={reward_text}",
            flush=True,
        )
```

**What changed:**
- Removed `tasks=`, `termination_reason=`, `delivery_score=`, `thermal_score=`, `efficiency_score=` — all of these break the validator's parser
- Changed `score={_fmt_float(end_score)}` (2 decimal places) to `score={score_clamped:.3f}` (3 decimal places — required by validator)
- Added `score_clamped = max(1e-6, min(end_score, 1 - 1e-6))` to prevent exact 0.0 or 1.0 values

---

## Bug 2 — Score clamped to `[0, 1]` instead of `(1e-6, 1-1e-6)`

### Location
`inference.py`, approximately line 465, inside the `task_outcomes.append(...)` block.

### Current broken code
```python
            task_outcomes.append(
                {
                    "task": task_id,
                    "score": max(0.0, min(1.0, task_score)),
                    "success": task_success,
                    "termination_reason": task_reason,
                    "steps": task_steps,
                }
            )
```

### Fixed code — change only the `"score"` line
```python
            task_outcomes.append(
                {
                    "task": task_id,
                    "score": max(1e-6, min(1 - 1e-6, task_score)),
                    "success": task_success,
                    "termination_reason": task_reason,
                    "steps": task_steps,
                }
            )
```

**What changed:** `max(0.0, min(1.0, ...))` → `max(1e-6, min(1 - 1e-6, ...))`. Scores of exactly 0.0 or 1.0 are rejected by the validator. This clamp ensures they never occur.

---

## Bug 3 — `end_score` also needs clamping at source

### Location
`inference.py`, approximately line 473, in the `if task_outcomes:` block.

### Current code
```python
        if task_outcomes:
            end_score = float(sum(item["score"] for item in task_outcomes) / len(task_outcomes))
            success = bool(all(bool(item["success"]) for item in task_outcomes))
            termination_reason = "all_tasks_completed"
```

### Fixed code — add one line after `end_score` is computed
```python
        if task_outcomes:
            end_score = float(sum(item["score"] for item in task_outcomes) / len(task_outcomes))
            end_score = max(1e-6, min(end_score, 1 - 1e-6))  # clamp before print
            success = bool(all(bool(item["success"]) for item in task_outcomes))
            termination_reason = "all_tasks_completed"
```

---

## Bug 4 — `[STEP]` line has extra `task=` field

### Location
`inference.py`, approximately lines 439–448, inside the step loop.

### Current code
```python
                print(
                    "[STEP] "
                    f"task={task_id} "
                    f"step={steps} "
                    f"action={_action_to_str(action)} "
                    f"reward={_fmt_float(reward)} "
                    f"done={_bool_text(done)} "
                    f"error={_format_error(env_error)}",
                    flush=True,
                )
```

### Fixed code — remove the `task=` field
```python
                print(
                    f"[STEP] "
                    f"step={steps} "
                    f"action={_action_to_str(action)} "
                    f"reward={_fmt_float(reward)} "
                    f"done={_bool_text(done)} "
                    f"error={_format_error(env_error)}",
                    flush=True,
                )
```

**Why:** The working reference repo's `[STEP]` format does not include `task=`. Extra fields in `[STEP]` may also confuse the output parser depending on the validator version.

---

## Verification — What the output must look like after fixes

Run `python inference.py --seed 42 --max-steps 20` locally and confirm stdout matches this pattern exactly:

```
[START] task=coldchain-gym env=coldchain-gym model=gpt-4.1-mini
[STEP] step=1 action=vehicle_index=0,action_type=1:REROUTE,target_index=3 reward=0.00 done=false error=null
[STEP] step=2 action=vehicle_index=0,action_type=1:REROUTE,target_index=5 reward=0.05 done=false error=null
...
[END] success=true steps=20 score=0.847 rewards=0.00,0.05,...
```

**Checklist before committing:**
- [ ] `[END]` line has exactly 4 fields: `success=`, `steps=`, `score=`, `rewards=`
- [ ] `score=` uses 3 decimal places (`0.847` not `0.85`)
- [ ] `score` value is between `0.001` and `0.999` (never exactly 0 or 1)
- [ ] No extra fields anywhere in `[END]` line
- [ ] `[STEP]` line has no `task=` field

---

## Bug 5 — `base.py` clamps score to `[0.0, 1.0]` inclusive (validator rejects exact boundaries)

### Location
`core/graders/base.py` — the `score()` method inside `BaseGrader`.

### Why this matters
Every grader (`BasicGrader`, `ModerateGrader`, `HardGrader`, `HardEmergencyCaseGrader`) inherits from `BaseGrader`. The clamp in `base.py` applies to all of them. The validator rejects scores of exactly `0.0` or `1.0` — they must be strictly inside the open interval `(1e-6, 1-1e-6)`.

`HardEmergencyCaseGrader._compute()` returns exactly `0.0` when cargo is destroyed and `1.0` when all conditions pass. After the current clamp these become exactly `0.0` and `1.0` → both rejected by the validator.

### Current broken code
```python
def score(self) -> float:
    if self._score is None:
        self._score = float(self._compute())
        self._score = max(0.0, min(1.0, self._score))          # ← accepts 0.0 and 1.0
        assert 0.0 <= self._score <= 1.0, f"{self.__class__.__name__} returned {self._score}, must be in [0,1]"
    return self._score
```

### Fixed code — change the clamp and assert lines only
```python
def score(self) -> float:
    if self._score is None:
        self._score = float(self._compute())
        self._score = max(1e-6, min(1 - 1e-6, self._score))    # ← strict open interval
        assert 1e-6 <= self._score <= 1 - 1e-6, f"{self.__class__.__name__} returned {self._score}, must be in (0,1)"
    return self._score
```

**What changed:** `max(0.0, min(1.0, ...))` → `max(1e-6, min(1 - 1e-6, ...))`. One line. Fixes all four graders at once since they all inherit this method.

---

## Pre-flight — Verify grader imports work before committing

Run this from your repo root after installing dependencies (`pip install networkx` if not already done):

```bash
python -c "
from core.graders.basic_grader import BasicGrader
from core.graders.moderate_grader import ModerateGrader
from core.graders.hard_grader import HardGrader
from core.graders.hard_emergency import HardEmergencyCaseGrader
print('BasicGrader:', BasicGrader.__name__)
print('ModerateGrader:', ModerateGrader.__name__)
print('HardGrader:', HardGrader.__name__)
print('HardEmergencyCaseGrader:', HardEmergencyCaseGrader.__name__)
print('ALL OK')
"
```

**Expected output:**
```
BasicGrader: BasicGrader
ModerateGrader: ModerateGrader
HardGrader: HardGrader
HardEmergencyCaseGrader: HardEmergencyCaseGrader
ALL OK
```

Must print `ALL OK` with no errors before committing. If `ModuleNotFoundError: No module named 'networkx'` appears, run `pip install networkx` first.

---

## Commit instructions

After all five fixes (`inference.py` × 4 + `base.py` × 1) and the pre-flight check passes:

```bash
git add inference.py core/graders/base.py
git commit -m "fix: conform [END] to OpenEnv spec, clamp score to (1e-6, 1-1e-6) in base grader"
git push origin Master
```

Then resubmit on the OpenEnv dashboard. Phase 2 Task Validation should pass.

---

## Do NOT change

- The `[START]` line — already correct
- The `openenv.yaml` — already fixed
- Any training or environment code
- The grader class names — they already match `openenv.yaml` entrypoints exactly
- `hard_emergency.py` — `HardEmergencyCaseGrader` returning `0.0`/`1.0` is fine; `base.py` clamp handles it