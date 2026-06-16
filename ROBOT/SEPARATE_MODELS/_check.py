import numpy as np, mujoco
from Env import ReachEnv, CarryEnv

e = ReachEnv()
e.reset()
_, r, _, _, info = e.step(np.zeros(6, dtype=np.float32))
bd = info['reward_breakdown']
gs_z = e.data.site_xpos[e.grasp_site_id][2]
op_z = e.data.qpos[e.object_qpos_addr + 2]
print("=== ReachEnv ===")
print(f"  gripper_z={gs_z:.3f}  cube_z={op_z:.3f}  z_gap={gs_z-op_z:.3f}")
print(f"  approach_z penalty : {bd['approach_z']:.4f}  (expect <0 if z_gap<0.03)")

e2 = CarryEnv()
e2.reset()
e2.data.qpos[e2.object_qpos_addr + 2] = 0.37
mujoco.mj_forward(e2.model, e2.data)
e2._best_obj_z = 0.28
_, r2, _, _, info2 = e2.step(np.zeros(6, dtype=np.float32))
bd2 = info2['reward_breakdown']
print("\n=== CarryEnv (cube at 0.37m) ===")
print(f"  over_height={bd2['over_height']:.4f}  (expect ~-0.45)")
print(f"  lift_dense ={bd2['lift_dense']:.4f}   (expect  0.0 — no reward above lift_height)")
print(f"  lift_pull  ={bd2['lift_pull']:.4f}    (expect  0.0 — above lift_height)")
print("\nOK")
