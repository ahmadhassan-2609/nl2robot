import mujoco
import numpy as np

# Target rotation matrix for all grasp/hover moves: gripper Z-axis pointing straight down.
#   R = [[1,  0,  0],
#        [0, -1,  0],    ← hand Z → world -Z (straight down)
#        [0,  0, -1]]
# This is a 180° rotation around the world X-axis.
TOP_DOWN_R = np.array([[1., 0., 0.],
                       [0.,-1., 0.],
                       [0., 0.,-1.]])


def _ori_error(R_current: np.ndarray, R_target: np.ndarray) -> np.ndarray:
    """
    Axis-angle orientation error from R_current to R_target.
    Returns a 3-vector (angular velocity direction × angle) in world frame.
    """
    R_err = R_target @ R_current.T
    cos_angle = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    if abs(angle) < 1e-7:
        return np.zeros(3)
    axis = np.array([R_err[2, 1] - R_err[1, 2],
                     R_err[0, 2] - R_err[2, 0],
                     R_err[1, 0] - R_err[0, 1]]) / (2.0 * np.sin(angle))
    return angle * axis


def get_joint_angles_for_pose(
    model,
    data,
    target_pos: np.ndarray,
    target_R: np.ndarray = TOP_DOWN_R,
    max_iter: int = 800,
    tol_pos: float = 1e-3,
    tol_ori: float = 1e-2,
) -> np.ndarray:
    """
    6-DOF Jacobian IK: drives the hand to target_pos with target_R orientation.

    Uses the full 6×7 Jacobian (jacp + jacr) so both position and orientation
    are controlled simultaneously.  Returns the resulting joint angles (first 7 DOF).
    """
    hand_id = model.body("hand").id

    for _ in range(max_iter):
        current_pos = data.body("hand").xpos.copy()
        pos_err = target_pos - current_pos
        ori_err = _ori_error(data.body("hand").xmat.reshape(3, 3), target_R)

        if np.linalg.norm(pos_err) < tol_pos and np.linalg.norm(ori_err) < tol_ori:
            break

        # Full 6×nv Jacobian (translational + rotational)
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, jacr, current_pos, hand_id)

        Jp = jacp[:, :7]
        Jr = jacr[:, :7]
        J = np.vstack([Jp, Jr])          # 6×7

        error_6d = np.concatenate([pos_err, ori_err])

        # Damped least-squares pseudoinverse step
        damping = 1e-3
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), error_6d)
        data.qpos[:7] += 0.03 * dq

        # Clamp to joint limits
        for i in range(7):
            jnt_id = model.joint(f"joint{i + 1}").id
            lo = model.jnt_range[jnt_id, 0]
            hi = model.jnt_range[jnt_id, 1]
            data.qpos[i] = np.clip(data.qpos[i], lo, hi)

        mujoco.mj_forward(model, data)

    return data.qpos[:7].copy()


def interpolate_trajectory(
    start: np.ndarray, end: np.ndarray, steps: int = 50
) -> list:
    """
    Joint-angle arrays smoothly interpolated from start to end.

    Uses a cosine ease-in / ease-out profile so joint velocity is zero at
    both endpoints and peaks in the middle of the move.  This eliminates the
    abrupt velocity step that a straight linspace produces at the start and
    end of every segment, giving all arm moves a natural acceleration and
    deceleration without requiring any additional infrastructure.
    """
    t_values = (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, steps))) / 2.0
    return [start + (end - start) * t for t in t_values]
