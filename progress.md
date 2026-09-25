# Progress Log

## 2026-09-23 Session
- v4 training line completed: Phases A/B/C have been run (A = 3 seeds × 50K samples + augmentation; B = authenticity/TTC head; C = 64ms window). Conclusion: architecture is the bottleneck; wind sense improved but direction degraded → stopped training iteration.
- v5 visual upgrade completed: brain panel / detailed effects / physics collision detection / precision attacks / manual sync; three-piece hashes all green.
- Design research completed: v4\full-fly-design.md (FlyGym / flybrain / OpenWorm / Neuro-LIFT research + 4-phase roadmap).
- New task initiated: fruit fly brain flies a drone (simulation) + GitHub project → task_plan.md (project root).
- Visual engine failure (image read failure 11+ times) → reminded user to run `npx @liustack/modlens doctor`.
- Drone simulation initiated and Phases 1–5 completed: drone_physics.js (hover drift 0.000) + drone_adapter.js (PID + mixing, escape response 64ms, pilot one-frame takeover) + continuous flight / obstacle avoidance / battery landing + smoke_drone.js 4/4 + GitHub packaging (LICENSE / .gitignore / bilingual README). Anchor hashes all green throughout. Remaining: Phase 6 demo GIF (optional), user visual verification.