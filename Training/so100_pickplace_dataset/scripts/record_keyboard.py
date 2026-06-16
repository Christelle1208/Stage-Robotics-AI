import os
from pathlib import Path

import mujoco
import numpy as np
import pygame
from lerobot.datasets.lerobot_dataset import LeRobotDataset


FPS = 30
EPISODES = int(os.getenv("EPISODES", "10"))
REPO_ID = os.getenv("REPO_ID", "local/so100_mujoco_pickplace_cartesian")
XML_PATH = os.getenv(
    "XML_PATH",
    "../so100-mujoco-sim/src/so100_mujoco_sim/xml/scene.xml",
)

TASK = "Pick the red cube and place it in the blue bin."

JOINT_NAMES = [
    "Rotation",
    "Pitch",
    "Elbow",
    "Wrist_Pitch",
    "Wrist_Roll",
    "Jaw",
]

EE_BODY_NAME = "Fixed_Jaw"


def get_joint_ids(model):
    ids = []
    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"Joint introuvable: {name}")
        ids.append(jid)
    return ids


def get_body_id(model, name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise RuntimeError(f"Body introuvable: {name}")
    return bid


def render_rgb(model, data, renderer):
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "side")
    if cam_id < 0:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front")

    if cam_id >= 0:
        renderer.update_scene(data, camera=cam_id)
    else:
        renderer.update_scene(data)

    return renderer.render().astype(np.uint8)


def get_joint_qpos(model, data, joint_ids):
    return np.array(
        [data.qpos[model.jnt_qposadr[jid]] for jid in joint_ids],
        dtype=np.float32,
    )


def set_joint_qpos(model, data, joint_ids, q):
    for i, jid in enumerate(joint_ids):
        data.qpos[model.jnt_qposadr[jid]] = q[i]


def clamp_joints(model, joint_ids, q):
    q = q.copy()
    for i, jid in enumerate(joint_ids):
        if model.jnt_limited[jid]:
            lo, hi = model.jnt_range[jid]
            q[i] = np.clip(q[i], lo, hi)
    return q


def get_rest_qpos(model, joint_ids):
    q = np.zeros(len(joint_ids), dtype=np.float32)

    for i, jid in enumerate(joint_ids):
        if model.jnt_limited[jid]:
            lo, hi = model.jnt_range[jid]
            q[i] = 0.5 * (lo + hi)
        else:
            q[i] = 0.0

    q[-1] = 0.0
    return q


def ik_step(model, data, joint_ids, ee_body_id, target_pos, q_current):
    mujoco.mj_forward(model, data)

    ee_pos = data.xpos[ee_body_id].copy()
    error = target_pos - ee_pos

    if np.linalg.norm(error) < 1e-4:
        return q_current

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, ee_body_id)

    dof_ids = []
    for jid in joint_ids[:5]:
        dof_ids.append(model.jnt_dofadr[jid])

    J = jacp[:, dof_ids]

    damping = 1e-3
    dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(3), error)

    dq = np.clip(dq, -0.04, 0.04)

    q_next = q_current.copy()
    q_next[:5] += dq
    q_next = clamp_joints(model, joint_ids, q_next)

    return q_next


def make_dataset():
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["joint"],
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["joint"],
        },
        "observation.images.front": {
            "dtype": "image",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channel"],
        },
    }

    return LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        robot_type="so100_mujoco",
        features=features,
        use_videos=True,
    )


def main():
    if not Path(XML_PATH).exists():
        raise FileNotFoundError(f"XML introuvable: {XML_PATH}")

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)

    joint_ids = get_joint_ids(model)
    ee_body_id = get_body_id(model, EE_BODY_NAME)

    dataset = make_dataset()

    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    pygame.display.set_caption("SO100 Cartesian Teleop")

    print("\nContrôle cartésien:")
    print("  W/S : avancer / reculer")
    print("  A/D : gauche / droite")
    print("  Q/E : monter / descendre")
    print("  R/F : tourner le poignet")
    print("  J/K : fermer / ouvrir pince")
    print("  SPACE : sauver épisode")
    print("  ESC   : quitter\n")

    for ep in range(EPISODES):
        mujoco.mj_resetData(model, data)

        q = get_rest_qpos(model, joint_ids)
        q = clamp_joints(model, joint_ids, q)
        set_joint_qpos(model, data, joint_ids, q)
        mujoco.mj_forward(model, data)

        target_pos = data.xpos[ee_body_id].copy()

        running = True
        clock = pygame.time.Clock()

        print(f"Épisode {ep + 1}/{EPISODES}")

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    dataset.save_episode()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        running = False
                    if event.key == pygame.K_ESCAPE:
                        dataset.save_episode()
                        return

            keys = pygame.key.get_pressed()

            delta = np.zeros(3, dtype=np.float32)
            step_xyz = 0.006

            if keys[pygame.K_w]:
                delta[0] += step_xyz
            if keys[pygame.K_s]:
                delta[0] -= step_xyz
            if keys[pygame.K_a]:
                delta[1] += step_xyz
            if keys[pygame.K_d]:
                delta[1] -= step_xyz
            if keys[pygame.K_q]:
                delta[2] += step_xyz
            if keys[pygame.K_e]:
                delta[2] -= step_xyz

            target_pos += delta

            target_pos[0] = np.clip(target_pos[0], 0.05, 0.30)
            target_pos[1] = np.clip(target_pos[1], -0.22, 0.22)
            target_pos[2] = np.clip(target_pos[2], 0.035, 0.30)

            q = ik_step(model, data, joint_ids, ee_body_id, target_pos, q)

            if keys[pygame.K_r]:
                q[4] += 0.035
            if keys[pygame.K_f]:
                q[4] -= 0.035

            if keys[pygame.K_j]:
                q[5] += 0.035
            if keys[pygame.K_k]:
                q[5] -= 0.035

            q = clamp_joints(model, joint_ids, q)

            set_joint_qpos(model, data, joint_ids, q)
            mujoco.mj_step(model, data)

            rgb = render_rgb(model, data, renderer)
            state = get_joint_qpos(model, data, joint_ids)

            dataset.add_frame(
                {
                    "observation.state": state,
                    "observation.images.front": rgb,
                    "action": q.astype(np.float32),
                    "task": TASK,
                }
            )

            surf = pygame.surfarray.make_surface(np.rot90(rgb))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            clock.tick(FPS)

        dataset.save_episode()
        print("  épisode sauvegardé")

    pygame.quit()
    print(f"Dataset terminé: {REPO_ID}")


if __name__ == "__main__":
    main()