import mujoco
import numpy as np


def get_scene_state(model, data) -> dict:
    """Returns current positions of all objects in the scene."""
    return {
        "red_block":    data.body("red_block").xpos.tolist(),
        "blue_block":   data.body("blue_block").xpos.tolist(),
        "green_block":  data.body("green_block").xpos.tolist(),
        "gripper":      data.body("hand").xpos.tolist(),
        "gripper_open": True,
    }


def get_object_position(data, object_name: str) -> np.ndarray:
    """Returns the world position of a named body."""
    return data.body(object_name).xpos.copy()
