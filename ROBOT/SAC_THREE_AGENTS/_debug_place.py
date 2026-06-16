from stable_baselines3 import SAC
from envs.place_env import PlaceEnv
import numpy as np

env = PlaceEnv()
model = SAC.load("models/best_place/best_model.zip")
obs, _ = env.reset(seed=0)

obj0 = env.data.qpos[env.object_qpos_addr:env.object_qpos_addr+3].copy()
jaw0 = float(env.data.qpos[5])
print(f"Start: obj={np.round(obj0,3)} jaw={jaw0:.3f} goal={np.round(env.goal_pos,3)}")
print(f"delta_scale={env.delta_scale}")
print()

for i in range(30):
    action, _ = model.predict(obs, deterministic=True)
    obs, rew, done, trunc, info = env.step(action)
    obj = env.data.qpos[env.object_qpos_addr:env.object_qpos_addr+3]
    jaw = float(env.data.qpos[5])
    bd = info["reward_breakdown"]
    print(f"step {i:3d}: act={np.round(action,2)}  obj={np.round(obj,3)}  jaw={jaw:.3f}  gdist={bd['goal_dist_m']:.3f}  held={bd['is_held']:.0f}  rew={rew:.3f}")
    if done or trunc:
        break
