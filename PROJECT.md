# Decomposition Learning for Robot Manipulation

## The Core Idea

Robot manipulation tasks have a natural structure that end-to-end methods ignore:

- **Geometric part**: Moving the arm to the right place. This can be solved analytically with motion planning — no learning needed.
- **Contact part**: What happens when the robot touches the object. This requires learning from interaction because contact physics (friction, compliance, misalignment) is hard to model.

End-to-end methods (like Diffusion Policy) learn both parts from scratch. This wastes data on the geometric part, and because the geometric part is robot-specific (joint angles, kinematics), the learned policy doesn't transfer to other robots.

**The claim**: Explicitly decomposing the task and only learning the contact part should give:
1. Better data efficiency (fewer demos needed)
2. Better cross-embodiment generalization (contact policy transfers to new robots)

## Task

**Peg-in-hole (PegInsertionSide-v1 in ManiSkill3)**

- Geometric part: Move peg to above the hole (~5cm). Done by motion planner.
- Contact part: Align and insert under positional uncertainty. Learned by policy.
- Contact detection: First frame where contact force > 0.1N

## Method

```
Demo data
    ↓
Split each trajectory at contact onset (force > 0.1N)
    ↓
Geometric part → replaced by motion planner at eval time
Contact part → train Diffusion Policy
    ↓
Policy input: relative position of TCP to hole (embodiment-agnostic)
Policy output: end-effector delta movements
```

The key design choice: policy input uses **relative position** (not joint angles), so it doesn't depend on the specific robot's kinematics. This is what enables cross-embodiment transfer.

## Experiments

### Experiment 1: Data Efficiency
- Train both methods with 50, 100, 200, 500, 996 demos
- Compare success rate curves
- Hypothesis: our method reaches 80% with fewer demos

### Experiment 2: Cross-Embodiment
- Train contact policy on Franka Panda demos
- Transfer directly to UR5 (zero-shot or with 5 demos fine-tuning)
- Baseline: retrain e2e from scratch on UR5
- Hypothesis: our contact policy transfers, e2e doesn't

## Baselines

1. **Diffusion Policy (e2e)**: Full task, state input (43-dim), same architecture
2. **PSL**: Motion planning + RL for contact (closest prior work, must beat this)
3. **SPIRE**: TAMP + IL for contact segments

