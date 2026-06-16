import numpy as np, mujoco
from Env import CarryEnv

e = CarryEnv()
e.reset()
# Simulate cube slipping (far from grasp_site)
e.data.qpos[e.object_qpos_addr:e.object_qpos_addr+3] = [0.5, -0.3, 0.28]
mujoco.mj_forward(e.model, e.data)
_, _, _, _, info = e.step(np.zeros(6, dtype=np.float32))
bd = info['reward_breakdown']
gs = e.data.site_xpos[e.grasp_site_id]
op = e.data.qpos[e.object_qpos_addr:e.object_qpos_addr+3]
grasp_d = float(np.linalg.norm(gs - op))
print(f"grasp_dist={grasp_d:.3f}  goal_dist={bd['goal_dist']:.4f}  (expect 0.0 — cube slipped)")

# Simulate high velocity
e2 = CarryEnv()
e2.reset()
e2.data.qvel[:6] = 3.0  # 3 rad/s on all joints
mujoco.mj_forward(e2.model, e2.data)
_, _, _, _, info2 = e2.step(np.zeros(6, dtype=np.float32))
bd2 = info2['reward_breakdown']
print(f"joint_vel penalty (qvel=3.0 x6): {bd2['joint_vel']:.4f}  (expect ~ -{0.3*6*9:.1f})")
print("OK")
