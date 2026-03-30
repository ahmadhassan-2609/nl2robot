# NL2Robot 🤖

### Natural Language Task Planner for Robotic Manipulation

**SPEC.md — Read this file before starting any work**

---

## Instructions for Claude Code

You are helping build NL2Robot from scratch. This file is the single source of truth for the entire project. Before writing any code:

1. Read this entire file
2. Create the full project folder structure as specified
3. Build each component in the order listed in the Build Order section
4. After each component is complete, confirm it works before moving to the next
5. If anything is unclear, ask before proceeding

---

## Project Overview

NL2Robot accepts a plain English command, uses an LLM to decompose it into a structured sequence of motion primitives, and executes it on a simulated Franka Panda robotic arm in MuJoCo.

**Example:**

```
Input:  "Pick up the red block and place it on top of the blue block"
Output: MuJoCo arm executing the task in real time
```

---

## Tech Stack

| Component | Tool |
| --- | --- |
| Simulation | MuJoCo 3.x |
| Robot Model | Franka Panda (MuJoCo Menagerie) |
| LLM Planner | Claude API (claude-sonnet-4-20250514) |
| Language | Python 3.10+ |

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   USER INPUT                        │
│         "Pick up red block, place on blue"          │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│                 SCENE MANAGER                       │
│  - Tracks object positions in real time             │
│  - Feeds current state to LLM as context            │
│  - Updates after each primitive executes            │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│                  LLM PLANNER                        │
│  - Receives: command + current scene state          │
│  - Reasons about task decomposition                 │
│  - Outputs: validated JSON action plan              │
│  - Handles: ambiguity, impossible tasks             │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│               PLAN VALIDATOR                        │
│  - Checks primitives exist                          │
│  - Checks objects referenced exist in scene         │
│  - Catches hallucinated actions before execution    │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│             PRIMITIVE EXECUTOR                      │
│  move_to / grasp / lift / place_on / release        │
│  - Translates abstract actions → MuJoCo controls    │
│  - Uses IK to compute joint angles                  │
│  - Executes with smooth trajectory interpolation    │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│              MUJOCO SIMULATION                      │
│  - Franka Panda 7-DOF arm                           │
│  - 3 colored blocks on tabletop                     │
│  - Real-time physics + rendering                    │
└─────────────────────────────────────────────────────┘
```

---

## Project Structure

```
nl2robot/
│
├── SPEC.md                  # This file
├── main.py                  # Entry point — run this
├── requirements.txt         # All dependencies
├── README.md                # Project readme
│
├── env/
│   ├── __init__.py
│   ├── scene.py             # MuJoCo env setup, object placement
│   ├── scene.xml            # MuJoCo XML — table, blocks, lighting
│   └── robot.py             # Franka model loader + viewer
│
├── planner/
│   ├── __init__.py
│   ├── llm_planner.py       # Claude API call + prompt engineering
│   ├── validator.py         # JSON plan validation
│   └── prompts.py           # System prompt + few-shot examples
│
├── executor/
│   ├── __init__.py
│   ├── primitives.py        # move_to, grasp, lift, place_on, release
│   └── controller.py        # IK solver + trajectory interpolation
│
└── utils/
    ├── __init__.py
    └── logger.py            # Pretty-prints plan + execution steps
```

---

## Build Order

Build in this exact sequence. Do not skip ahead.

1. Project structure and requirements.txt
2. env/scene.xml — MuJoCo scene with table and blocks
3. env/scene.py — scene loader and object position tracker
4. executor/controller.py — IK solver and trajectory interpolation
5. executor/primitives.py — all 6 motion primitives
6. planner/prompts.py — system prompt
7. planner/llm_planner.py — Claude API call
8. planner/validator.py — plan validation
9. utils/logger.py — pretty printer
10. main.py — wire everything together
11. Test with first command, debug until working
12. README.md — write after everything works

---

## Component Specifications

### 1. requirements.txt

```
mujoco>=3.0.0
anthropic>=0.20.0
numpy>=1.24.0
```

---

### 2. env/scene.xml

The scene contains:

- A flat table in the center
- 3 colored blocks: red, blue, green (each 4cm cube with free joints)
- Overhead lighting
- The Franka Panda arm will be included from MuJoCo Menagerie

```xml
<mujoco model="nl2robot_scene">
  <option gravity="0 0 -9.81"/>

  <worldbody>
    <!-- Table -->
    <body name="table" pos="0.5 0 0.4">
      <geom type="box" size="0.4 0.4 0.02" rgba="0.8 0.7 0.6 1"/>
    </body>

    <!-- Red Block -->
    <body name="red_block" pos="0.4 0.1 0.44">
      <joint type="free"/>
      <geom type="box" size="0.02 0.02 0.02" rgba="1 0 0 1" mass="0.1"/>
    </body>

    <!-- Blue Block -->
    <body name="blue_block" pos="0.4 -0.1 0.44">
      <joint type="free"/>
      <geom type="box" size="0.02 0.02 0.02" rgba="0 0 1 1" mass="0.1"/>
    </body>

    <!-- Green Block -->
    <body name="green_block" pos="0.5 0.0 0.44">
      <joint type="free"/>
      <geom type="box" size="0.02 0.02 0.02" rgba="0 1 0 1" mass="0.1"/>
    </body>
  </worldbody>
</mujoco>
```

---

### 3. env/scene.py

```python
def get_scene_state(model, data):
    """Returns current positions of all objects in the scene."""
    return {
        "red_block":   data.body("red_block").xpos.tolist(),
        "blue_block":  data.body("blue_block").xpos.tolist(),
        "green_block": data.body("green_block").xpos.tolist(),
        "gripper":     data.body("hand").xpos.tolist(),
        "gripper_open": True
    }
```

---

### 4. executor/controller.py

Numerical IK using Jacobian pseudoinverse. Iterates until the end-effector reaches the target or max iterations are hit. Also includes smooth linear trajectory interpolation between joint configurations.

```python
import mujoco
import numpy as np

def get_joint_angles_for_position(model, data, target_pos: np.ndarray) -> np.ndarray:
    max_iter = 100
    tol = 1e-3
    for _ in range(max_iter):
        current_pos = data.body("hand").xpos.copy()
        error = target_pos - current_pos
        if np.linalg.norm(error) < tol:
            break
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, None, target_pos,
                      model.body("hand").id)
        J = jacp[:, :7]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), error)
        data.qpos[:7] += 0.1 * dq
        mujoco.mj_forward(model, data)
    return data.qpos[:7].copy()

def interpolate_trajectory(start: np.ndarray, end: np.ndarray,
                           steps: int = 50) -> list:
    return [start + (end - start) * t for t in np.linspace(0, 1, steps)]
```

---

### 5. executor/primitives.py

6 primitives. Each is self-contained and testable independently.

| Primitive | Arguments | Description |
| --- | --- | --- |
| `move_to` | `object: str` | Move gripper to hover position above object |
| `grasp` | `object: str` | Lower and close gripper around object |
| `lift` | `height: float` | Lift gripper straight up by height in meters |
| `place_on` | `object: str` | Move above target and lower onto it |
| `release` | none | Open gripper fingers |
| `move_home` | none | Return arm to neutral configuration |

Constants to use:

```python
HOVER_HEIGHT = 0.12   # meters above object before grasping
GRASP_OFFSET = 0.02   # meters — final approach distance
HOME_CONFIG  = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
```

Implement as a class `PrimitiveExecutor` that takes `model`, `data`, and optional `viewer` in its constructor. Each method should print what it's doing (e.g. `→ move_to(red_block)`) and call `mujoco.mj_step` + `viewer.sync()` after each interpolation step.

---

### 6. planner/prompts.py

```python
SYSTEM_PROMPT = """
You are a robot task planner. Your job is to decompose a natural language
command into a sequence of motion primitives that a robot arm can execute.

## Available Primitives
- move_to(object): Move gripper above a named object
- grasp(object): Close gripper to pick up object
- lift(height): Lift gripper by height in meters (max 0.3)
- place_on(object): Move above target object and lower onto it
- release(): Open gripper
- move_home(): Return to neutral position

## Objects in the current scene
{scene_state}

## Rules
1. Always move_to before grasp
2. Always grasp before lift
3. Always lift before place_on
4. Always release after placing
5. Always end with move_home
6. If the command is impossible or ambiguous, set "feasible": false
   and explain why in "reason"

## Output Format
Return ONLY valid JSON. No explanation outside the JSON.

{
  "feasible": true,
  "reasoning": "step by step explanation of your plan",
  "steps": [
    {"action": "move_to",  "args": {"object": "red_block"}},
    {"action": "grasp",    "args": {"object": "red_block"}},
    {"action": "lift",     "args": {"height": 0.15}},
    {"action": "place_on", "args": {"object": "blue_block"}},
    {"action": "release",  "args": {}},
    {"action": "move_home","args": {}}
  ]
}
"""
```

---

### 7. planner/llm_planner.py

```python
import anthropic
import json
import re
from planner.prompts import SYSTEM_PROMPT

client = anthropic.Anthropic()

def plan_task(command: str, scene_state: dict) -> dict:
    prompt = SYSTEM_PROMPT.format(scene_state=json.dumps(scene_state, indent=2))
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=prompt,
        messages=[{"role": "user", "content": command}]
    )
    raw = response.content[0].text
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"LLM returned invalid JSON: {raw}")
```

---

### 8. planner/validator.py

```python
VALID_ACTIONS  = {"move_to", "grasp", "lift", "place_on", "release", "move_home"}
VALID_OBJECTS  = {"red_block", "blue_block", "green_block"}

def validate_plan(plan: dict) -> tuple[bool, str]:
    if not plan.get("feasible", False):
        return False, f"LLM marked task infeasible: {plan.get('reason', 'no reason given')}"
    if "steps" not in plan or not plan["steps"]:
        return False, "Plan contains no steps"
    for i, step in enumerate(plan["steps"]):
        action = step.get("action")
        args   = step.get("args", {})
        if action not in VALID_ACTIONS:
            return False, f"Step {i}: Unknown action '{action}'"
        if "object" in args and args["object"] not in VALID_OBJECTS:
            return False, f"Step {i}: Unknown object '{args['object']}'"
        if action == "lift":
            h = args.get("height", 0)
            if not (0 < h <= 0.3):
                return False, f"Step {i}: Lift height {h} out of safe range (0, 0.3]"
    return True, "OK"
```

---

### 9. utils/logger.py

Pretty-prints the LLM's reasoning and action plan to the terminal before execution. Each step should be numbered and clearly formatted.

---

### 10. main.py

```python
import mujoco
import mujoco.viewer
from env.scene import get_scene_state
from planner.llm_planner import plan_task
from planner.validator import validate_plan
from executor.primitives import PrimitiveExecutor
from utils.logger import log_plan

def run(command: str):
    model = mujoco.MjModel.from_xml_path("env/scene.xml")
    data  = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    print(f"\n🗣  Command: \"{command}\"")
    print("─" * 50)

    scene_state = get_scene_state(model, data)
    print(f"📍 Scene: {scene_state}")

    print("\n🧠 Planning with LLM...")
    plan = plan_task(command, scene_state)

    is_valid, msg = validate_plan(plan)
    if not is_valid:
        print(f"❌ Plan invalid: {msg}")
        return

    log_plan(plan)

    print("\n🤖 Executing in MuJoCo...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        executor = PrimitiveExecutor(model, data, viewer)
        action_map = {
            "move_to":   lambda args: executor.move_to(args["object"]),
            "grasp":     lambda args: executor.grasp(args["object"]),
            "lift":      lambda args: executor.lift(args["height"]),
            "place_on":  lambda args: executor.place_on(args["object"]),
            "release":   lambda args: executor.release(),
            "move_home": lambda args: executor.move_home(),
        }
        for step in plan["steps"]:
            action_map[step["action"]](step.get("args", {}))

    print("\n✅ Task complete.")

if __name__ == "__main__":
    commands = [
        "Pick up the red block and place it on the blue block",
        "Move the green block next to the red block",
        "Stack all three blocks on top of each other",
    ]
    for cmd in commands:
        run(cmd)
```

---

## Known Challenges & Fixes

| Challenge | Fix |
| --- | --- |
| IK doesn't converge | Add damping term to pseudoinverse, increase iterations |
| LLM returns text outside JSON | Use regex to extract JSON block (already handled) |
| Franka model path issues | Use absolute path in XML include |
| Gripper doesn't close on object | Adjust GRASP_OFFSET constant |
| Viewer crashes on exit | Wrap in try/finally to close cleanly |
| MuJoCo version conflicts | Pin to mujoco==3.1.6 in requirements.txt |
| `venv\Scripts\activate` fails on Windows | Run `Set-ExecutionPolicy RemoteSigned` in PowerShell first |

---

## README.md (write this last)

```markdown
# NL2Robot 🤖

Natural language task planning for robotic manipulation using LLMs + MuJoCo.

## What it does
Type a plain English command → LLM decomposes it into motion primitives →
Franka Panda arm executes it in MuJoCo simulation.

## Demo
"Pick up the red block and place it on the blue block"
[insert GIF here]

## Architecture
User Input → Scene Manager → LLM Planner (Claude) →
Plan Validator → Primitive Executor → MuJoCo

## Setup
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
python main.py

## Why this matters
LLMs as high-level robot planners is one of the most active research
directions in robotics right now (SayCan, Code as Policies, RT-2).
This project implements the core idea from scratch in a clean,
demonstrable system.
```

---

## Stretch Goals (if time allows)

- [ ]  Feedback loop — if execution fails, re-plan with updated scene state
- [ ]  Support multi-step stacking: *"Stack all three blocks"*
- [ ]  Simple text input GUI instead of hardcoded commands
- [ ]  Record a GIF of the demo for the README

---

*Built with Claude Code — NL2Robot, 2026*