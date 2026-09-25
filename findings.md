# Findings & Decision Log (Drone Simulation + GitHub)

## Decided Decisions
1. **Not using Rapier physics engine**: the package is ESM-only / requires a bundler, violating the file:// zero-build constraint (verified against official README) → hand-written dynamics.
2. **Layered control** (referenced from v4\full-fly-design.md): fruit fly brain = decision (pseudo-muscle commands), PID attitude loop = stability (biological equivalent: VNC / haltere reflex), pilot remote control = highest priority.
3. **SNN metrics anchor**: results.json / jitter_sim_log.md hashes are locked; must re-verify after any phase change.
4. **GitHub license: MIT** (code); data references FlyWire (Dorkenwald et al., Nature 634:124–138, 2024); note "Google open source" = Neuroglancer viewer, data = FlyWire/Princeton.

## Reference Resources
- snedea/flybrain (MIT): engineering reference for real-time whole-brain 139K neuron browser simulation (Web Worker LIF + WebGL panel).
- heyseth/worm-sim (MIT): predecessor project, 302-neuron C. elegans browser simulation.
- PX4 offboard / gym-pybullet-drones: control stack reference for future real-hardware phase.
- Training findings (v4 experiments): data augmentation fixes wind sense (0→55% trigger) but direction accuracy degrades from 10.7° to 19°; bottleneck is model architecture, not data — GitHub project uses v3 model (10.7° / 90.3%).

## Risks
- world.html change regression risk → run smoke_test + hash verification every phase.
- Flight dynamics numerical stability (integration divergence at large dt) → semi-implicit Euler + dt clamping.