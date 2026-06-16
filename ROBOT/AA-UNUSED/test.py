import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
import mujoco
import time
import numpy as np

# Charger le modèle SmolVLA
model_id = "lerobot/smolvla_base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = SmolVLAPolicy.from_pretrained(model_id).to(device).eval()
preprocess, postprocess = make_pre_post_processors(
    policy.config,
    model_id,
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)

# Charger le modèle MuJoCo
model = mujoco.MjModel.from_xml_path("mujoco_menagerie/trs_so_arm100/scene_with_cube_and_cameras.xml")
data = mujoco.MjData(model)

# Instruction textuelle et état initial
task_text = "avoid red cube"
init_state = np.array([1, 0.8, 0.2, 0.2, 0.2, 1.5], dtype=np.float32)  # Exemple d'état initial (6 dimensions pour 6 joints)
state = torch.from_numpy(init_state).reshape(1, 1, -1)

# Boucle de simulation avec visualisation
with mujoco.viewer.launch_passive(model, data) as viewer:
    num_steps = 450
    for step in range(num_steps):
        # Image factice au bon format et type
        image = torch.ones((1, 3, 256, 256), dtype=torch.uint8)
        frame = {
            "observation.images.camera1": image,
            "observation.state": state,
            "observation.language.text": task_text,
            "task": task_text,  # Ajout de la clé attendue
        }
        batch = preprocess(frame)
        with torch.inference_mode():
            action = policy.select_action(batch)
        data.ctrl[:] = action[:model.nu] if hasattr(action, '__len__') else action
        for _ in range(10):
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)
        #print(f"Étape {step+1}")
