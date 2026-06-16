import mujoco
import mujoco.viewer
import numpy as np
import time
# from lerobot.datasets.lerobot_dataset import LeRobotDataset
# from lerobot.policies.factory import make_pre_post_processors
# from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# Charger le modèle MuJoCo
model = mujoco.MjModel.from_xml_path("mujoco_menagerie/trs_so_arm100/scene_pick_place.xml")
data = mujoco.MjData(model)

# Rendu offscreen pour capturer les images (256x256 RGB)
renderer = mujoco.Renderer(model, height=256, width=256)

# Configuration
num_episodes = 200
dataset = []
viewer_pause = 0.5  # secondes d'affichage par épisode

with mujoco.viewer.launch_passive(model, data) as viewer:
  viewer.cam.azimuth = 135
  viewer.cam.elevation = -20
  viewer.cam.distance = 1.2

  # Joints du bras uniquement (exclure les free joints, ex. cube)
  arm_jnt_ids = [i for i in range(model.njnt)
                 if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
  arm_dof_ids  = [model.jnt_dofadr[j] for j in arm_jnt_ids]

  for episode in range(num_episodes):
    # Reset simulation
    mujoco.mj_resetData(model, data)

    # Random initialization — bras uniquement (le cube garde sa pose par défaut)
    for i in arm_jnt_ids:
        qpos_idx = model.jnt_qposadr[i]
        data.qpos[qpos_idx] = np.random.uniform(
            model.jnt_range[i, 0],
            model.jnt_range[i, 1]
        )
    mujoco.mj_forward(model, data)
    viewer.sync()

    # Capture de l'image d'observation (état initial)
    renderer.update_scene(data, camera="track_cam")
    obs_image = renderer.render().copy()  # (256, 256, 3) uint8

    # Sauvegarder le qpos initial du bras (6 joints)
    initial_arm_qpos = data.qpos[:6].copy()

    # Position du cube
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_pos = data.xpos[cube_id].copy()
    print(f"Episode {episode + 1} | Cube position: {cube_pos}")

    # IK pour amener le milieu de la pince sur le cube
    fixed_jaw_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Fixed_Jaw")
    moving_jaw_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Moving_Jaw")
    target_pos = cube_pos
    lam = 0.01  # facteur d'amortissement (damped least squares)

    for _ in range(500):
        mujoco.mj_forward(model, data)
        # Milieu géométrique entre les deux mâchoires
        gripper_center = (data.xpos[fixed_jaw_id] + data.xpos[moving_jaw_id]) / 2.0
        error = target_pos - gripper_center

        if np.linalg.norm(error) < 0.01:
            break

        # Jacobien = moyenne des deux mâchoires
        jacp_fixed  = np.zeros((3, model.nv))
        jacp_moving = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp_fixed,  None, fixed_jaw_id)
        mujoco.mj_jacBody(model, data, jacp_moving, None, moving_jaw_id)
        jacp_full = (jacp_fixed + jacp_moving) / 2.0
        jacp_arm = jacp_full[:, arm_dof_ids]  # (3, n_arm_dofs)

        # Moindres carrés amortis : dq = J^T (J J^T + λ²I)⁻¹ e
        dq_arm = jacp_arm.T @ np.linalg.solve(
            jacp_arm @ jacp_arm.T + lam**2 * np.eye(3), error
        )

        # Appliquer et clipper sur les limites articulaires
        for k, j in enumerate(arm_jnt_ids):
            qpos_idx = model.jnt_qposadr[j]
            data.qpos[qpos_idx] = np.clip(
                data.qpos[qpos_idx] + dq_arm[k],
                model.jnt_range[j, 0],
                model.jnt_range[j, 1]
            )
        viewer.sync()

    # Afficher la pose finale et marquer une pause
    mujoco.mj_forward(model, data)
    viewer.sync()
    gripper_center = (data.xpos[fixed_jaw_id] + data.xpos[moving_jaw_id]) / 2.0
    err_final = np.linalg.norm(target_pos - gripper_center)
    print(f"Episode {episode + 1} | erreur IK finale: {err_final:.4f} m")

    if err_final < 0.02:
        # Capture de l'image goal (état final, pince proche du cube)
        renderer.update_scene(data, camera="track_cam")
        goal_image = renderer.render().copy()  # (256, 256, 3) uint8

        # Action = delta qpos bras (initial → final)
        final_arm_qpos = data.qpos[:6].copy()
        action = final_arm_qpos - initial_arm_qpos

        dataset.append({
            "obs_image":    obs_image,
            "goal_image":   goal_image,
            "initial_qpos": initial_arm_qpos,
            "final_qpos":   final_arm_qpos,
            "action":       action,
            "cube_position": cube_pos,
        })
        print(f"  → ajouté au dataset ({len(dataset)} épisodes)")

    time.sleep(viewer_pause)

print(f"Dataset créé avec {len(dataset)} épisodes")

# Sauvegarde en .npz pour finetune_octo.py
if dataset:
    np.savez(
        "so100_dataset.npz",
        obs_images   = np.array([ep["obs_image"]    for ep in dataset], dtype=np.uint8),
        goal_images  = np.array([ep["goal_image"]   for ep in dataset], dtype=np.uint8),
        actions      = np.array([ep["action"]       for ep in dataset], dtype=np.float32),
        initial_qpos = np.array([ep["initial_qpos"] for ep in dataset], dtype=np.float32),
        final_qpos   = np.array([ep["final_qpos"]   for ep in dataset], dtype=np.float32),
    )
    print("Dataset sauvegardé dans so100_dataset.npz")