# NL2Robot

**Natural language task planning for robotic manipulation — built from scratch using Claude + MuJoCo.**

Type a plain English command. A 7-DOF Franka Panda arm executes it in real-time physics simulation.

```
"Pick up the red block and place it on the blue block"
```

---

## What It Does

NL2Robot bridges the gap between language and physical robot control. It takes an unstructured natural language command, uses an LLM to decompose it into a structured motion plan, validates that plan against physical and semantic constraints, then executes it on a simulated robot arm — with smooth trajectories, stable block stacking, and no manual scripting.

![Animation](https://github.com/user-attachments/assets/ce04606c-2938-4f4e-a083-a96a77960a9c)

The entire pipeline runs end-to-end from a single Python invocation. No ROS. No training data. No learned motion policies.

---

## Demo

```
Command: "Pick up the red block and place it on the blue block"
──────────────────────────────────────────────────────────────
Scene state: {
  red_block:   [0.40,  0.10, 0.44],
  blue_block:  [0.40, -0.10, 0.44],
  green_block: [0.50,  0.00, 0.44],
  gripper:     [0.45,  0.00, 0.65]
}

Planning with LLM (Claude)...

==================================================
  ROBOT PLAN
==================================================

Reasoning:
  Pick up red_block using move_to, grasp, and lift, then
  place it on top of blue_block and return to home.

Steps (6 total):
   1. ->      move_to(object=red_block)
   2. [grip]  grasp(object=red_block)
   3. ^       lift(height=0.15)
   4. v       place_on(object=blue_block)
   5. [open]  release()
   6. [home]  move_home()
==================================================

Executing in MuJoCo...
  -> move_to(red_block)       gripper @ [0.400, 0.100, 0.560]
  -> grasp(red_block)         gripper @ [0.400, 0.100, 0.500]
  -> lift(height=0.15)        gripper @ [0.400, 0.100, 0.650]
  -> place_on(blue_block)     gripper @ [0.400,-0.100, 0.524]
  -> release()                gripper @ [0.400,-0.100, 0.524]
  -> move_home()              gripper @ [0.451, 0.000, 0.650]
```

---

## Architecture

```
┌─────────────────────────────────────────────┐
│              Natural Language Input         │
│   "Pick up the red block and place it on    │
│    the blue block"                          │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│               Scene Manager                 │
│  Reads body positions from MuJoCo xpos      │
│  Serializes world state to JSON for LLM     │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│             LLM Planner (Claude)            │
│  Receives: command + grounded scene state   │
│  Outputs:  validated JSON action sequence   │
│  Handles:  ambiguity, infeasible tasks      │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│               Plan Validator                │
│  Checks: action names, object existence,    │
│  parameter bounds — deterministically,      │
│  without involving the LLM                  │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│            Primitive Executor               │
│                                             │
│  move_to  →  chained Jacobian IK            │
│  grasp    →  approach + close fingers       │
│  lift     →  straight-up Cartesian move     │
│  place_on →  hover + precision lower        │
│  release  →  locked-arm finger ramp         │
│  move_home → two-phase retract + sweep      │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│            MuJoCo 3.x Simulation            │
│  Franka Panda 7-DOF arm                     │
│  3 colored blocks, physics-stable stacking  │
│  Real-time passive viewer                   │
└─────────────────────────────────────────────┘
```

---

## Technical Design

### LLM as a Semantic Parser

The LLM (Claude Sonnet) receives the current world state as structured JSON and the command as free text. It outputs a JSON action plan — nothing more. It never touches joint angles, trajectories, or physics. This is a deliberate separation of concerns: the LLM reasons about *what* to do; the executor handles *how* to do it.

The primitive vocabulary acts as a constrained grammar. Because the LLM can only produce combinations of 6 known actions on known objects, the space of possible plans is small and verifiable. The `feasible` flag lets the model explicitly refuse ambiguous or impossible tasks rather than producing a wrong-but-valid-looking plan.

### Jacobian IK with Chained Waypoints

Inverse kinematics is solved numerically using damped least-squares over the full 6×7 Jacobian (3 translational + 3 rotational rows):

```
dq = Jᵀ (J Jᵀ + λI)⁻¹ · e₆ᴅ
```

A naive single-endpoint IK solve can converge to a different posture branch than the robot is currently in — producing what looks like a sudden jump to a reset configuration. To prevent this, `_move_to_pos` divides every Cartesian move into `N=5` intermediate waypoints, solving IK at each one seeded from the previous result. Because consecutive Cartesian positions are small steps, consecutive joint configurations stay in the same branch.

All IK solving runs on a private `mujoco.MjData` copy so the passive viewer's background render thread never sees intermediate IK states.

### Kinematic Control + Block Teleportation

The arm is controlled kinematically — `data.qpos[:7]` is written directly each step rather than through actuators. This eliminates PID tuning, gravity compensation, and torque limits from the problem.

The trade-off: MuJoCo never generates contact forces large enough to lift a block kinematically. The solution is `_carry_block`: at grasp time, the block's position relative to the hand is recorded in hand-local frame. Every subsequent step, the block's free-joint qpos is recomputed from the current hand transform and teleported — perfectly mimicking rigid attachment.

### Physics Stability

Stack stability required careful tuning:

- `timestep=0.001` (half the default) for numerical stability under stiff contacts
- `solimp="0.99 0.999 0.001"` + `solref="0.004 1"` on block geoms — very stiff contact model that prevents inter-penetration drift
- Global `viscosity=0.05` to damp small oscillations
- 200-step gripper open ramp with arm joints locked throughout release — prevents finger contact impulses from disturbing the placed stack
- 300-step settling period before the arm departs

### Trajectory Smoothing

All joint-space interpolation uses a cosine ease-in/ease-out profile:

```python
t = (1 - cos(π·i/n)) / 2
```

Joint velocity is zero at both endpoints. Every motion — hover, grasp approach, lift, placement, home return — accelerates smoothly from rest and decelerates to rest, with no velocity discontinuities at waypoint boundaries.

---

## Relation to Prior Work

This project implements the core idea from two influential robotics papers:

**SayCan (Ahn et al., 2022)** grounds LLM planning in physical affordances — the LLM only selects actions the robot is actually capable of executing. NL2Robot achieves this through the fixed primitive vocabulary and the plan validator, which hard-gates execution on physical feasibility.

**Code as Policies (Liang et al., 2023)** uses LLMs to write robot code directly. NL2Robot takes the complementary approach: the LLM outputs a structured plan in a constrained DSL (the JSON action schema) rather than freeform code. This makes the planning output auditable and safe to validate programmatically before any motion begins.

The key insight both papers share — and which this project demonstrates concretely — is that LLMs are strong semantic reasoners but poor low-level controllers. The right architecture uses LLMs at the task level and classical methods (IK, trajectory interpolation, physics simulation) at the motion level.

---

## Project Structure

```
nl2robot/
├── main.py                  # Entry point
├── requirements.txt
│
├── env/
│   ├── scene.xml            # MuJoCo world: table, blocks, Panda arm
│   └── scene.py             # Scene state reader
│
├── planner/
│   ├── llm_planner.py       # Claude API call
│   ├── prompts.py           # System prompt + primitive schema
│   └── validator.py         # Deterministic plan safety check
│
├── executor/
│   ├── primitives.py        # 6 motion primitives + PrimitiveExecutor
│   └── controller.py        # Jacobian IK + cosine trajectory interpolation
│
└── utils/
    └── logger.py            # Plan pretty-printer
```

---

## Setup

**Requirements:** Python 3.10+, an Anthropic API key.

```bash
git clone <repo>
cd nl2robot
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...   # or set in .env
```

**Run the default demo:**

```bash
python main.py
```

**Run a custom command:**

```bash
python main.py "Pick up the green block and place it on the red block"
```

---

## Example Commands

```bash
python main.py "Pick up the red block and place it on the blue block"
python main.py "Move the green block on top of the red block"
python main.py "Stack the red block on the blue block"
python main.py "Put the blue block next to the green block"
python main.py "Can you move the blocks"       # LLM marks infeasible: ambiguous
python main.py "Pick up the purple block"      # Validator rejects: unknown object
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `mujoco` | 3.1.6 | Physics simulation + rendering |
| `anthropic` | ≥0.20.0 | Claude API client |
| `numpy` | ≥1.24.0 | Jacobian IK, trajectory math |

---

*Built with Claude Code.*
