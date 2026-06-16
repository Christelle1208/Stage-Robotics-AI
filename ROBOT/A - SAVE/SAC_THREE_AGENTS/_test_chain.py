"""Quick smoke-test: verify all three envs and state injection chain."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from envs import ReachEnv, GraspEnv, PlaceEnv

reach = ReachEnv()
grasp = GraspEnv()
place = PlaceEnv()

# Run 10 steps in reach, capture snapshot
obs, _ = reach.reset(seed=42)
for _ in range(10):
    obs, r, done, trunc, info = reach.step(reach.action_space.sample())
qpos, qctrl = reach.get_state_snapshot()
print(f"Reach snapshot: qpos={qpos.shape}, qctrl={qctrl.shape}")

# Inject into grasp
obs, _ = grasp.reset_from_reach(qpos, qctrl)
print(f"GraspEnv after inject: obs={obs.shape}")
for _ in range(5):
    obs, r, done, trunc, info = grasp.step(grasp.action_space.sample())
qpos2, qctrl2 = grasp.get_state_snapshot()

# Inject into place
obs, _ = place.reset_from_grasp(qpos2, qctrl2)
print(f"PlaceEnv after inject: obs={obs.shape}")
for _ in range(5):
    obs, r, done, trunc, info = place.step(place.action_space.sample())

print("State injection chain: PASS")
reach.close()
grasp.close()
place.close()
