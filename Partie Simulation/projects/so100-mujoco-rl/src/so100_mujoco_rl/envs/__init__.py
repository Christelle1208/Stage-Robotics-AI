from so100_mujoco_rl.envs.so100_reach_env import SO100ReachEnv  # noqa: F401
from so100_mujoco_rl.envs.so100_pick_place_env import SO100PickPlaceEnv  # noqa: F401
from so100_mujoco_rl.envs.so100_real_pick_place_env import SO100RealPickPlaceEnv  # noqa: F401
from so100_mujoco_rl.envs.so100_grab_env import SO100GrabEnv  # noqa: F401

# Register both environments with Gymnasium so gym.make() works.
import gymnasium as gym  # noqa: E402

if "SO100Reach-v0" not in gym.envs.registry:
    gym.register(
        id="SO100Reach-v0",
        entry_point="so100_mujoco_rl.envs:SO100ReachEnv",
        max_episode_steps=200,
    )

if "SO100PickPlace-v0" not in gym.envs.registry:
    gym.register(
        id="SO100PickPlace-v0",
        entry_point="so100_mujoco_rl.envs:SO100PickPlaceEnv",
        max_episode_steps=200,
    )

# Real-life reproduction: table + 5 cm cube + 10×10×5 cm bin.
# Uses SO100RealPickPlaceEnv which builds the scene programmatically.
if "SO100RealPickPlace-v0" not in gym.envs.registry:
    gym.register(
        id="SO100RealPickPlace-v0",
        entry_point="so100_mujoco_rl.envs:SO100RealPickPlaceEnv",
        max_episode_steps=300,
    )

# Reach-and-grab on the Feuille scene: cube spawns on the checkerboard,
# success = end-effector near the cube with the gripper closed.
if "SO100Grab-v0" not in gym.envs.registry:
    gym.register(
        id="SO100Grab-v0",
        entry_point="so100_mujoco_rl.envs:SO100GrabEnv",
        max_episode_steps=150,
    )
