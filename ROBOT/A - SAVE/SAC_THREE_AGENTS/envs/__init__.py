"""
SO-100 SAC Three-Agent environments.

Import the task environments:
  ReachEnv       — Agent 1: move end-effector to pre-grasp position (SAC)
  ScriptedGrasp  — Agent 2: close gripper (scripted, no model)
  PlaceEnv       — Agent 3: transport to goal position and release (SAC)
  BoxPlaceEnv    — Agent 3 variant: drop cube into an open box (SAC)
"""

from envs.reach_env import ReachEnv
from envs.grasp_env import ScriptedGrasp
from envs.place_env import PlaceEnv
from envs.box_place_env import BoxPlaceEnv

# Backward-compatible alias
GraspEnv = ScriptedGrasp

__all__ = ["ReachEnv", "ScriptedGrasp", "GraspEnv", "PlaceEnv", "BoxPlaceEnv"]
