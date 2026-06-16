"""
Agent 2 — GRASP (Scripted Controller — no SAC model needed)

The Grasp phase is a simple scripted action: close the gripper jaw
incrementally over a fixed number of steps, then let physics settle.

This replaces the SAC-based GraspEnv because the Reach agent already
positions the EE precisely above the object with the gripper open.
All that remains is to close the jaw — no learning required.

Interface:
  ScriptedGrasp.run(reach_env) → returns (qpos, qctrl, success)
  Can also be used as a Gymnasium env for API compatibility.
"""

import numpy as np
import mujoco

from envs.base_env import SO100BaseEnv, CONFIG


class ScriptedGrasp:
    """
    Scripted gripper closure — replaces GraspEnv SAC agent.

    Usage:
        grasp = ScriptedGrasp(render_mode="human")
        qpos, qctrl, success = grasp.run(reach_env)
    """

    def __init__(self, render_mode=None):
        gc = CONFIG["grasp"]
        self.close_steps    = gc["close_steps"]
        self.delta_per_step = gc["delta_per_step"]
        self.settle_steps   = gc["settle_steps"]
        self.lift_height    = gc["success"]["lift_height"]

        self.gripper_threshold = CONFIG["robot"]["gripper_cube_jaw"]

        self.render_mode = render_mode

    def run(self, source_env: SO100BaseEnv) -> tuple[np.ndarray, np.ndarray, bool]:
        """
        Close the gripper on the object using the physics state from source_env.

        Args:
            source_env: environment whose MuJoCo state contains the
                        post-Reach configuration (EE above object, jaw open).

        Returns:
            (qpos, qctrl, success):
                qpos/qctrl — simulator state after grasping (for PlaceEnv)
                success    — True if object is held (jaw closed around it)
        """
        model = source_env.model
        data  = source_env.data
        frame_skip = source_env.frame_skip

        total_steps = 0

        # Phase 1: gradually close the jaw
        for _ in range(self.close_steps):
            # Decrement jaw ctrl toward 0 (closed)
            data.ctrl[5] = max(0.0, data.ctrl[5] - self.delta_per_step)
            mujoco.mj_step(model, data, frame_skip)
            total_steps += 1

            if self.render_mode == "human" and source_env._viewer is not None:
                source_env._viewer.sync()

        # Phase 2: let physics settle (object may shift during grip)
        for _ in range(self.settle_steps):
            mujoco.mj_step(model, data, frame_skip)
            total_steps += 1

            if self.render_mode == "human" and source_env._viewer is not None:
                source_env._viewer.sync()

        # Check success: jaw closed around object
        jaw_angle = float(data.qpos[5])
        success = jaw_angle < self.gripper_threshold

        return data.qpos.copy(), data.ctrl.copy(), success


# Keep GraspEnv as an alias for API compatibility (imports, evaluate.py, etc.)
GraspEnv = ScriptedGrasp
