import sys
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
    mujoco.mj_forward(model, data)  # propagate qpos -> xpos before reading state

    print(f"\n Command: \"{command}\"")
    print("-" * 50)

    scene_state = get_scene_state(model, data)
    print(f"Scene state: {scene_state}")

    print("\nPlanning with LLM (Claude)...")
    plan = plan_task(command, scene_state)

    is_valid, msg = validate_plan(plan)
    if not is_valid:
        print(f"Plan invalid: {msg}")
        return

    log_plan(plan)

    print("\nExecuting in MuJoCo...")
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Front view: camera in front of the table looking toward the robot
            viewer.cam.azimuth   = 180    # 180 = camera at +Y, looking toward -Y
            viewer.cam.elevation = -20    # tilt down slightly
            viewer.cam.distance  =  2.0
            viewer.cam.lookat[0] =  0.25  # centre between robot (x=0) and table (x=0.5)
            viewer.cam.lookat[1] =  0.0
            viewer.cam.lookat[2] =  0.55  # mid-height of arm workspace
            viewer.sync()
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

            # Keep viewer open until user closes the window.
            # Freeze arm joints so gravity doesn't collapse the arm,
            # but still step physics so the placed block can settle.
            print("\nExecution complete. Close the MuJoCo window to exit.")
            final_arm_qpos = data.qpos[:9].copy()  # 7 arm joints + 2 fingers
            while viewer.is_running():
                mujoco.mj_step(model, data)
                data.qpos[:9] = final_arm_qpos
                data.qvel[:9] = 0.0
                mujoco.mj_forward(model, data)
                viewer.sync()

    except Exception as e:
        print(f"Execution error: {e}")
        raise

    print("\nTask complete.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Allow passing a command directly: python main.py "Pick up the red block"
        run(" ".join(sys.argv[1:]))
    else:
        commands = [
            "Pick up the red block and place it on the blue block",
        ]
        for cmd in commands:
            run(cmd)
