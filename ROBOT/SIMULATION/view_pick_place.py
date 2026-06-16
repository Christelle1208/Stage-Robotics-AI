"""
view_pick_place.py
------------------
Ouvre simplement la scène pick-and-place du SO-100 dans un viewer MuJoCo.

Lancer avec :
    mjpython view_pick_place.py
"""

import mujoco
import mujoco.viewer
import numpy as np
import time

# SCENE = "mujoco_menagerie/trs_so_arm100/scene_pick_place.xml"
SCENE = "trs_so_arm100/TEST_SCENE.xml"
SCENE = "/Users/christelle.nollet/Desktop/ROBOT/SAC_THREE_AGENTS/assets/scene_table.xml"

model = mujoco.MjModel.from_xml_path(SCENE)
data  = mujoco.MjData(model)

# Pose initiale : bras étendu horizontalement, gripper ouvert
INIT_QPOS = np.array([0.0, -1.57, 0.05, 0.0, 0.0, -0.174], dtype=np.float32)
data.qpos[:6] = INIT_QPOS
data.ctrl[:] = INIT_QPOS
mujoco.mj_forward(model, data)

print("Scène chargée :", SCENE)
print("Fermer la fenêtre ou Ctrl+C pour quitter.")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.004)   # ~250 Hz, même timestep que la scène
