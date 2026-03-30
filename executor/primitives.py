import time
import mujoco
import numpy as np

from env.scene import get_object_position
from executor.controller import get_joint_angles_for_pose, interpolate_trajectory

HOVER_HEIGHT = 0.12   # hand height above block center for hover (fingers clear block top)
# With top-down orientation the fingers are 0.058 m below the hand.
# GRASP_OFFSET = 0.06  -> finger centres sit at block centre (block_z + 0.002)
GRASP_OFFSET = 0.06
# Reach-ready home: hand at ~[0.40, 0.00, 0.65], gripper Z pointing straight down
HOME_CONFIG  = np.array([1.9088, -0.274, -1.9756, -2.5453, -0.4954, 2.602, 1.1471])

# Workspace floor constraint
# Table surface = table body pos z (0.40) + box half-height (0.02) = 0.42 world Z.
# The end-effector must stay at least EE_Z_MARGIN above that surface so the
# fingers and wrist links never clip through the table top.
TABLE_Z      = 0.42   # world Z of table surface
EE_Z_MARGIN  = 0.05   # 5 cm safety clearance above table
EE_Z_MIN     = TABLE_Z + EE_Z_MARGIN   # = 0.47 m

# Number of intermediate Cartesian waypoints used for chained IK in _move_to_pos.
# Higher values keep the arm closer to the Cartesian straight line and make
# it harder for the IK to jump to a different posture branch.
_N_IK_WAYPOINTS = 5


class PrimitiveExecutor:
    def __init__(self, model, data, viewer=None):
        self.model  = model
        self.data   = data
        self.viewer = viewer
        self._gripper_open   = True
        self._held_block     = None   # name of currently grasped block
        self._grasp_offset_L = None   # block offset in hand-local frame at grasp time

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _step_sim(self):
        """Step physics, sync viewer, and sleep so movement is visible."""
        mujoco.mj_step(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()
        time.sleep(0.02)

    def _print_gripper_pos(self):
        pos = self.data.body("hand").xpos
        print(f"     gripper @ [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    def _carry_block(self):
        """If a block is held, teleport its free-joint qpos to follow the hand.

        The arm is driven kinematically (qpos set directly), so MuJoCo never
        generates the contact forces needed to lift the block.  We compensate by
        recomputing the block world position from the current hand transform and
        the offset recorded at grasp time.
        """
        if self._held_block is None:
            return
        hand_xpos = self.data.body("hand").xpos.copy()
        hand_R    = self.data.body("hand").xmat.reshape(3, 3)
        new_pos   = hand_xpos + hand_R @ self._grasp_offset_L
        adr = int(np.asarray(self.model.joint(f"{self._held_block}_joint").qposadr).flat[0])
        self.data.qpos[adr    :adr + 3] = new_pos
        self.data.qpos[adr + 3]         = 1.0   # w — keep block upright
        self.data.qpos[adr + 4:adr + 7] = 0.0   # x y z
        mujoco.mj_forward(self.model, self.data)

    def _execute_trajectory(self, waypoints: list, steps_per_waypoint: int = 50):
        """
        Drive the arm through a list of (start, end) joint-angle pairs.
        Each element of `waypoints` is already a target qpos array (7 DOF).
        """
        current = self.data.qpos[:7].copy()
        for target in waypoints:
            traj = interpolate_trajectory(current, target, steps=steps_per_waypoint)
            for q in traj:
                self.data.qpos[:7] = q
                self._step_sim()
                self._carry_block()
            current = target

    def _move_to_pos(self, target_pos: np.ndarray, steps: int = 50):
        """Chained IK through intermediate Cartesian waypoints.

        Splits the straight-line Cartesian path from current hand position to
        target_pos into _N_IK_WAYPOINTS segments.  Each IK solve is seeded from
        the previous result, so consecutive joint configurations stay close in
        joint space and the arm never jumps to a different posture branch.
        """
        target_pos = target_pos.copy()
        if target_pos[2] < EE_Z_MIN:
            target_pos[2] = EE_Z_MIN

        start_pos = self.data.body("hand").xpos.copy()

        # Private IK sandbox — live data is never modified during solving.
        ik_data = mujoco.MjData(self.model)
        ik_data.qpos[:] = self.data.qpos[:]
        mujoco.mj_forward(self.model, ik_data)

        # Cosine-spaced t values so waypoints are denser near start/end.
        t_vals = (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, _N_IK_WAYPOINTS + 1))) / 2.0

        # Solve chained IK; each solve seeds from the previous result.
        ik_configs = [self.data.qpos[:7].copy()]
        for t in t_vals[1:]:
            wp = start_pos + (target_pos - start_pos) * t
            wp[2] = max(wp[2], EE_Z_MIN)
            q = get_joint_angles_for_pose(self.model, ik_data, wp)
            ik_configs.append(q)

        # Drive the arm through consecutive IK configs with cosine easing.
        steps_per_seg = max(3, steps // _N_IK_WAYPOINTS)
        for i in range(len(ik_configs) - 1):
            q_a, q_b = ik_configs[i], ik_configs[i + 1]
            t_ease = (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, steps_per_seg + 1))) / 2.0
            for t in t_ease[1:]:
                self.data.qpos[:7] = q_a + (q_b - q_a) * t
                self._step_sim()
                self._carry_block()

    def _set_gripper(self, open_val: float):
        """
        Set finger joint positions.
        open_val=0.04 -> open, open_val=0.0 -> closed.
        """
        finger_indices = [
            self.model.joint("finger_joint1").qposadr,
            self.model.joint("finger_joint2").qposadr,
        ]
        for idx in finger_indices:
            self.data.qpos[idx] = open_val
        mujoco.mj_forward(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

    # ------------------------------------------------------------------
    # Primitives
    # ------------------------------------------------------------------

    def move_to(self, object_name: str):
        """Move gripper to hover position above the named object."""
        print(f"  -> move_to({object_name})")
        obj_pos = get_object_position(self.data, object_name)
        hover_pos = obj_pos.copy()
        hover_pos[2] += HOVER_HEIGHT
        self._move_to_pos(hover_pos)
        self._print_gripper_pos()

    def grasp(self, object_name: str):
        """Lower gripper to object and close fingers."""
        print(f"  -> grasp({object_name})")
        obj_pos = get_object_position(self.data, object_name)
        grasp_pos = obj_pos.copy()
        grasp_pos[2] += GRASP_OFFSET

        # Open gripper before approach
        self._set_gripper(0.04)

        # Lower to grasp position
        self._move_to_pos(grasp_pos, steps=30)

        # Close gripper
        for val in np.linspace(0.04, 0.0, 20):
            self._set_gripper(val)
            self._step_sim()

        self._gripper_open = False

        # Record block offset in hand-local frame so _carry_block can track it
        hand_xpos = self.data.body("hand").xpos.copy()
        hand_R    = self.data.body("hand").xmat.reshape(3, 3)
        block_pos = self.data.body(object_name).xpos.copy()
        self._grasp_offset_L = hand_R.T @ (block_pos - hand_xpos)
        self._held_block     = object_name

        self._print_gripper_pos()

    def lift(self, height: float):
        """Lift the gripper straight up by `height` meters."""
        print(f"  -> lift(height={height})")
        current_pos = self.data.body("hand").xpos.copy()
        target_pos  = current_pos.copy()
        target_pos[2] += height
        self._move_to_pos(target_pos)
        self._print_gripper_pos()

    def place_on(self, object_name: str):
        """Move above target object and lower the gripper onto it."""
        print(f"  -> place_on({object_name})")
        obj_pos = get_object_position(self.data, object_name)

        # Move above target
        above_pos = obj_pos.copy()
        above_pos[2] += HOVER_HEIGHT
        self._move_to_pos(above_pos)

        # Compute exact placement height so the held block sits just 2 mm above
        # the target's top surface.  With top-down orientation hand-Z = world -Z,
        # so block_z = hand_z - grasp_offset_L[2].  Solving for hand_z:
        #   desired block_z = target_z + 2*half_height + 0.002 gap
        #   hand_z          = desired_block_z + grasp_offset_L[2]
        BLOCK_HALF = 0.02
        if self._grasp_offset_L is not None:
            desired_block_z = obj_pos[2] + 2 * BLOCK_HALF + 0.002
            place_z = desired_block_z + self._grasp_offset_L[2]
        else:
            place_z = obj_pos[2] + 0.10  # fallback

        place_pos = obj_pos.copy()
        place_pos[2] = place_z
        self._move_to_pos(place_pos, steps=30)
        self._print_gripper_pos()

    def release(self):
        """Open gripper fingers slowly so no sudden impulse destabilises the stack."""
        print("  -> release()")
        # Lock arm joint angles for the entire release so contact forces from the
        # opening fingers cannot nudge the arm and disturb the placed block.
        locked_arm_qpos = self.data.qpos[:7].copy()

        # Drop tracking so the block is free from this point on
        self._held_block     = None
        self._grasp_offset_L = None

        # Ramp open over 200 physics steps (~0.2 s sim-time at dt=0.001).
        # Bypasses the 0.02 s wall-clock sleep so the ramp is fast to watch
        # while still being gradual in simulation time.
        for val in np.linspace(0.0, 0.04, 200):
            self._set_gripper(val)
            self.data.qpos[:7] = locked_arm_qpos   # hold arm position
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None:
                self.viewer.sync()

        # Allow 300 settling steps (~0.3 s sim-time) for the stack to stabilise
        # before the arm moves away, without adding wall-clock delay.
        for _ in range(300):
            self.data.qpos[:7] = locked_arm_qpos   # hold arm position
            mujoco.mj_step(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()
        self._gripper_open = True
        self._print_gripper_pos()

    def move_home(self):
        """Return arm to neutral (home) configuration."""
        print("  -> move_home()")
        # Phase 1 — retract straight up from the current position.
        # Lifting clear of the placed stack before swinging away makes the
        # departure look deliberate rather than a sudden lateral snap.
        current_pos = self.data.body("hand").xpos.copy()
        retract_pos = current_pos.copy()
        retract_pos[2] += HOVER_HEIGHT          # rise one hover-height above current
        self._move_to_pos(retract_pos, steps=50)
        # Phase 2 — sweep to home with enough steps for a gradual arc.
        # 160 steps at dt=0.001 keeps the same wall-clock feel as before
        # while giving the cosine profile room to ease in and out visibly.
        self._execute_trajectory([HOME_CONFIG], steps_per_waypoint=160)
        self._print_gripper_pos()
