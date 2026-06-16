"""Visualise the Place env standalone reset state."""
import mujoco
import mujoco.viewer
from envs.place_env import PlaceEnv

env = PlaceEnv(render_mode="human")
obs, _ = env.reset(seed=42)

# Print state
obj = env.data.qpos[env.object_qpos_addr:env.object_qpos_addr + 3]
ee = env.data.site_xpos[env.grasp_site_id]
jaw = float(env.data.qpos[5])
print(f"EE:  {ee}")
print(f"Obj: {obj}")
print(f"Goal: {env.goal_pos}")
print(f"Jaw: {jaw:.3f}")

viewer = mujoco.viewer.launch_passive(env.model, env.data)
print("Viewer open — close the window to exit.")
while viewer.is_running():
    mujoco.mj_step(env.model, env.data)
    viewer.sync()
viewer.close()
