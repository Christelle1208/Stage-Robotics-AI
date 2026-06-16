
import mujoco

import mujoco
import mujoco.viewer
import numpy as np
import time

# POUR RUN LE CODE UTILISER mypython simu_test.py

# # --- Ajout LeRobot + SmolVLA ---
# import torch
# from lerobot.datasets.lerobot_dataset import LeRobotDataset
# from lerobot.policies.factory import make_pre_post_processors
# from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# # Charger le modèle MuJoCo
# model = mujoco.MjModel.from_xml_path("mujoco_menagerie/trs_so_arm100/scene.xml")
# data = mujoco.MjData(model)

# # Charger la policy SmolVLA
# model_id = "lerobot/smolvla_base"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# policy = SmolVLAPolicy.from_pretrained(model_id).to(device).eval()
# preprocess, postprocess = make_pre_post_processors(
#     policy.config,
#     model_id,
#     device=device,
#     preprocessor_overrides={"device_processor": {"device": str(device)}},
# )

# # Charger le dataset LeRobot
# dataset = LeRobotDataset("lerobot/libero")
# episode_index = 0
# from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]
# to_idx   = dataset.meta.episodes["dataset_to_index"][episode_index]
# print("IDX range for episode:", from_idx, "to", to_idx)
#         # Ajout clé factice pour le langage si absente
#         import torch
#         if "observation.language.tokens" not in frame:
#             frame["observation.language.tokens"] = torch.zeros((1, 1), dtype=torch.long)
#         if "observation.language.attention_mask" not in frame:
#             frame["observation.language.attention_mask"] = torch.ones((1, 1), dtype=torch.bool)
#         else:
#             frame["observation.language.attention_mask"] = frame["observation.language.attention_mask"].bool()
#         # Ajout clé factice pour l'état si absent ou mal formaté
#         if "observation.state" not in frame or not hasattr(frame["observation.state"], 'shape') or len(frame["observation.state"].shape) < 2:
#             # Adapte la taille selon ce que la policy attend (exemple : batch=1, seq_len=1, state_dim=10)
#             frame["observation.state"] = torch.zeros((1, 1, 10), dtype=torch.float32)
#         # Si besoin, ajouter d'autres mappings ici
#         batch = preprocess(frame)
#         with torch.inference_mode():
#             pred_action = policy.select_action(frame)
#             pred_action = postprocess(pred_action)
#         # Appliquer l'action à la simulation
#         data.ctrl[:] = np.array(pred_action[:model.nu])
#         mujoco.mj_step(model, data)
#         viewer.sync()
#         if step % 100 == 0:
#             print(f"Step {step}, Joint positions: {data.qpos}")
#         time.sleep(0.01)

# --- Section interactive pour envoyer des ordres aux moteurs ---       FONCTIONNELLE
model = mujoco.MjModel.from_xml_path("mujoco_menagerie/trs_so_arm100/scene_pick_place.xml")
data = mujoco.MjData(model)


print("\n=== Contrôle manuel des moteurs MuJoCo ===")
print(f"Ce robot possède {model.nu} moteurs.\n")
# Affiche les limites des joints
print("Limites des joints :")
for i in range(model.nu):
	min_lim = model.jnt_range[i, 0] if hasattr(model, 'jnt_range') else None
	max_lim = model.jnt_range[i, 1] if hasattr(model, 'jnt_range') else None
	print(f"  Moteur {i}: min={min_lim}, max={max_lim}")
# print("\nEntrez les positions cibles des moteurs sous forme de liste, ex : [0.1, -0.2, ...] ou 'exit' pour quitter.")

with mujoco.viewer.launch_passive(model, data) as viewer:
	while True:
		try:
			user_input = input("Entrez les positions cibles : ")
			if user_input.strip().lower() in ["exit", "quit", "q"]:
				print("Arrêt du contrôle manuel.")
				break
			# Évalue l'entrée utilisateur comme une liste
			target = eval(user_input, {"__builtins__": None}, {})
			target = np.array(target, dtype=np.float32)
			if target.shape[0] != model.nu:
				print(f"Erreur : il faut {model.nu} valeurs.")
				continue
			# Appliquer la commande
			data.ctrl[:] = target
			# Avancer la simulation sur plusieurs pas pour rendre le mouvement visible
			for _ in range(400):
				mujoco.mj_step(model, data)
				viewer.sync()
				time.sleep(0.01)
			print(f"Commande envoyée : {target}")
			print(f"Position réelle des joints : {data.qpos}")
		except Exception as e:
			print(f"Entrée invalide ({e}). Essayez encore.")
   
   
# # --- Actions aléatoires parmi les limites des moteurs ---
# model = mujoco.MjModel.from_xml_path("mujoco_menagerie/trs_so_arm100/scene.xml")
# data = mujoco.MjData(model)

# print("\n=== Actions aléatoires dans les limites des moteurs ===")
# with mujoco.viewer.launch_passive(model, data) as viewer:
# 	for step in range(1000):
# 		# Générer une action aléatoire dans les limites des moteurs
# 		target = np.array([np.random.uniform(model.jnt_range[i, 0], model.jnt_range[i, 1]) for i in range(model.nu)], dtype=np.float32)
# 		# Appliquer la commande
# 		data.ctrl[:] = target
# 		# Avancer la simulation
# 		mujoco.mj_step(model, data)
# 		viewer.sync()
# 		time.sleep(0.01)
# 		if step % 100 == 0:
# 			print(f"Étape {step}, Commande : {target}")
# 			# print(f"Position réelle des joints : {data.qpos}")