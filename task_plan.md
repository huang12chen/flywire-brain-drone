# Task Plan: Fruit Fly Brain Flies a Drone (Simulation) + GitHub Open Source Project

> Goal: upgrade the current "fruit fly brain = escape decision" to "fruit fly brain continuously pilots a drone flight (3D simulation)",
> and package it as a publishable GitHub project. The v4\ training plan has been completed (see v4\task_plan.md, archived for history).

## Goal Statement
1. Simulation version of "fruit fly brain flies a drone": brain stays online continuously — cruising, obstacle avoidance, escape, landing; drone has real flight dynamics (mass/lift/drag/attitude), not teleportation.
2. GitHub project: README (bilingual + demo GIF) + LICENSE + reproducibility instructions + GitHub Pages online demo.
3. Iron rules unchanged: SNN metrics unchanged (web\results.json SHA256=`0E1339A0…71E9E`, jitter `5C66B3D0…072AE`); pilot takeover has highest priority at all times; file:// double-click usable.

## Current Status (verified 2026-09-23)
- web\world.html:286 `brain: new SNNRuntime(...)`; :466 driveAgent; :512 `escapeVec = Vector3(o0,o2,o1)` (y/z mapping done); :526-528 `vel += escapeVec*accel*dtMs` (acceleration direct push, no dynamics).
- Existing: connectome brain (1871 neurons / 45524 edges), 3D world, cross-platform swap, escape decision, brain panel (brain_panel.js), detailed effects (scene_fx.js), precision attacks, pilot takeover.
- Gaps: ① No flight dynamics (no mass/lift/attitude) ② No adaptation layer (brain → motor) ③ Brain only outputs during threats (no cruise mode) ④ No GitHub packaging.

## Phases
| # | Content | Deliverables | Status |
|---|---------|-------------|--------|
| 1 | Flight dynamics drone_physics.js: mass/lift/drag/gravity/attitude quaternion integration; 4 motor speeds → thrust + torque; hover/acceleration/attitude response self-tests | drone_physics.js + self-test script | pending |
| 2 | Control adaptation layer drone_adapter.js: GF burst → throttle step, escape direction → desired tilt angle → PID attitude loop → mixing → 4 motors; pilot takeover bypass preserved | drone_adapter.js | pending |
| 3 | Continuous flight: cruise (slow wandering), obstacle avoidance (LPLC2 continuous sensing → steering), escape (existing), landing; HUD adds motor speed / attitude angle | world.html update | pending |
| 4 | Regression: smoke_test expansion (hover stability / escape response / takeover latency) + three-piece hash anchors + world.html syntax | Regression record | pending |
| 5 | GitHub packaging: LICENSE (MIT), .gitignore, bilingual README (architecture diagram + metrics table + demo GIF placeholder), reproducibility instructions, Pages deployment guide | Repo ready | pending |
| 6 | (Optional) Demo GIF / screen recording + promotional copy | GIF / copy | pending |

## Honest Boundaries (README must state)
- Cruise / obstacle avoidance "steering preference" is a programmatic baseline + neural escape hybrid, not fully neural-generated (full neural cruise = complete fruit fly route Phase 2, see v4\full-fly-design.md).
- Flight dynamics parameters are demo-grade (typical small quadcopter values), not calibrated for real hardware.
- Brain output is an "escape direction vector"; the adaptation layer translates it into motor commands (biological analogy: brain → thoracic ganglion → flight muscle).

## Errors Encountered
| Error | Attempt Count | Resolution |
|-------|---------------|------------|
| (To be recorded) | | |