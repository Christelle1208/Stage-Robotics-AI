"""
BENCHMARK ZERO-SHOT - Version finale pour votre SO100
======================================================

Configuration automatique basée sur vos fichiers XML:
- so_arm100.xml (modèle du bras)
- scene_pick_place.xml (scène avec cube et cible)

Prêt à l'emploi !
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
from typing import Dict, List, Tuple, Optional, Any
import time
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from PIL import Image

# ============================================================================
# CONFIGURATION - ADAPTÉE À VOS FICHIERS XML
# ============================================================================

# Chemin vers votre scène (qui inclut le bras via <include>)
SCENE_XML_PATH = "SIMULATION/trs_so_arm100/TEST_SCENE.xml"  # ← Votre fichier de scène

# Configuration des modèles à utiliser
USE_REAL_MODELS = {
    "smolvla": False,  # True = utiliser le vrai SmolVLA, False = version simulée
    "octo": True,     # True = utiliser le vrai Octo, False = version simulée
    "sac": False,      # True = charger un checkpoint SAC, False = actions aléatoires
}

# Chemins vers les checkpoints (si USE_REAL_MODELS = True)
MODEL_CHECKPOINTS = {
    "sac": "SIMULATION/outputs/reach_sac/best_model.zip",       # Chemin vers le checkpoint SAC
    "sac_vecnorm": "SIMULATION/outputs/reach_sac/vecnorm.pkl",  # VecNormalize indispensable
}

# Paramètres du benchmark
BENCHMARK_CONFIG = {
    "num_episodes": 5,      # Nombre d'épisodes par modèle
    "render": False,        # Afficher la simulation (ralentit)
    "save_videos": False,   # Sauvegarder des vidéos des épisodes
    "verbose": True,        # Affichage détaillé
    "log_actions_every": 25,  # Afficher les actions prédites tous les N steps (0 = désactivé)
}

# Configuration de la tâche pick-and-place
# Positions extraites de votre scene_pick_place.xml
TASK_CONFIG = {
    "object_initial_pos": np.array([0.00, -0.31, 0.018]),  # Position du cube
    "target_pos": np.array([0.18, -0.08, 0.08]),            # Position du goal_site
    "grasp_threshold": 0.04,     # Distance pour considérer l'objet saisi
    "placement_threshold": 0.05,  # Distance pour placement réussi
}

# Noms des éléments dans vos XML (déjà identifiés)
XML_ELEMENT_NAMES = {
    "end_effector_body": "Fixed_Jaw",     # End-effector = mâchoire fixe
    "gripper_body": "Moving_Jaw",         # Gripper = mâchoire mobile
    "object_body": "cube",                # Objet à manipuler
    "target_site": "goal_site",           # Site cible (pas un body)
    "gripper_joint": "Jaw",               # Joint du gripper
    "camera": "track_cam",                # Caméra de tracking
}

# Informations sur les joints (extraites de so_arm100.xml)
ROBOT_JOINTS = {
    "names": ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"],
    "home_qpos": [0, -1.57, 1.57, 1.57, -1.57, 0],  # Position home
    "num_actuators": 6,  # 6 actuateurs
}

# Limites réelles des actuateurs SO100 (en radians)
JOINT_LIMITS = {
    "low":  np.array([-1.92, -3.32, -0.174, -1.66, -2.79, -0.174], dtype=np.float32),
    "high": np.array([ 1.92,  0.174,  3.14,  1.66,  2.79,  1.75 ], dtype=np.float32),
}

# ============================================================================
# MÉTRIQUES ET DATACLASSES
# ============================================================================

@dataclass
class EpisodeMetrics:
    """Métriques pour un épisode"""
    success: bool
    steps: int
    total_reward: float
    grasp_success: bool
    placement_success: bool
    execution_time: float
    model_name: str
    task_name: str
    episode_id: int


# ============================================================================
# ENVIRONNEMENT MUJOCO POUR SO100
# ============================================================================

class SO100PickPlaceEnv(gym.Env):
    """
    Environnement MuJoCo pour le SO100 avec tâche pick-and-place
    Utilise vos fichiers XML scene_pick_place.xml et so_arm100.xml
    """
    
    def __init__(
        self, 
        xml_path: str,
        render_mode: Optional[str] = None,
        task_config: Optional[Dict] = None,
        element_names: Optional[Dict] = None,
        robot_info: Optional[Dict] = None
    ):
        super().__init__()
        
        self.xml_path = xml_path
        self.render_mode = render_mode
        self.task_config = task_config or TASK_CONFIG
        self.element_names = element_names or XML_ELEMENT_NAMES
        self.robot_info = robot_info or ROBOT_JOINTS
        
        # Charger votre modèle SO100
        print(f"📂 Chargement de la scène depuis: {xml_path}")
        try:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
            self.data = mujoco.MjData(self.model)
            print(f"✅ Modèle chargé:")
            print(f"   • DoF (nq): {self.model.nq}")
            print(f"   • Actuateurs (nu): {self.model.nu}")
            print(f"   • Corps: {self.model.nbody}")
        except Exception as e:
            raise ValueError(f"Impossible de charger {xml_path}: {e}")
        
        # Identifier les indices importants
        self._identify_model_elements()
        
        # Définir les espaces d'observation et d'action
        # Action space = vraies limites articulaires (en radians)
        self.action_space = spaces.Box(
            low=JOINT_LIMITS["low"],
            high=JOINT_LIMITS["high"],
            dtype=np.float32
        )
        
        # Observation simplifiée pour les modèles
        # Format: [qpos_robot (6), qvel_robot (6), cube_pos (3), goal_pos (3), ee_pos (3)]
        obs_dim = 6 + 6 + 3 + 3 + 3  # 21 dimensions
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        self.viewer = None
        self.max_episode_steps = 250
        self.current_step = 0

        # Indices des éléments du modèle — initialisés ici pour éviter AttributeError
        # si _identify_model_elements() ne les trouve pas
        self.ee_body_id = None
        self.object_body_id = None
        self.object_qpos_start = None
        self.target_site_id = None
        self.robot_joint_indices = []

        # Variables de tâche
        self.object_grasped = False
        self.initial_object_pos = None
        self.target_pos = None
        
        # Renderer pour les observations visuelles
        self.camera_renderer = None
        
    def _identify_model_elements(self):
        """Identifie les éléments importants dans vos XML"""
        print("\n🔍 Identification des éléments du modèle...")
        
        # Trouver l'end-effector (Fixed_Jaw)
        try:
            self.ee_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, 
                self.element_names["end_effector_body"]
            )
            print(f"  ✅ End-effector trouvé: {self.element_names['end_effector_body']}")
        except:
            print(f"  ⚠️  End-effector '{self.element_names['end_effector_body']}' non trouvé")
            self.ee_body_id = None
        
        # Trouver le cube (objet)
        try:
            self.object_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY,
                self.element_names["object_body"]
            )
            
            # Trouver l'index qpos du cube (joint free)
            for i in range(self.model.njnt):
                joint_type = self.model.jnt_type[i]
                if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                    body_id = self.model.jnt_bodyid[i]
                    if body_id == self.object_body_id:
                        self.object_qpos_start = self.model.jnt_qposadr[i]
                        print(f"  ✅ Objet trouvé: {self.element_names['object_body']} (qpos idx: {self.object_qpos_start})")
                        break
        except Exception as e:
            print(f"  ⚠️  Objet '{self.element_names['object_body']}' non trouvé: {e}")
            self.object_body_id = None
            self.object_qpos_start = None
        
        # Trouver la cible (goal_site est un site, pas un body)
        try:
            self.target_site_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE,
                self.element_names["target_site"]
            )
            print(f"  ✅ Cible trouvée: {self.element_names['target_site']} (site)")
        except:
            print(f"  ⚠️  Cible '{self.element_names['target_site']}' non trouvée")
            self.target_site_id = None
        
        # Identifier les indices des joints du robot
        self.robot_joint_indices = []
        for joint_name in self.robot_info["names"]:
            try:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                qpos_idx = self.model.jnt_qposadr[joint_id]
                self.robot_joint_indices.append(qpos_idx)
            except:
                print(f"  ⚠️  Joint '{joint_name}' non trouvé")
        
        print(f"  ✅ Joints du robot identifiés: {len(self.robot_joint_indices)}")
        
        # Caméra
        try:
            cam_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA,
                self.element_names["camera"]
            )
            print(f"  ✅ Caméra trouvée: {self.element_names['camera']}")
        except:
            print(f"  ⚠️  Caméra '{self.element_names['camera']}' non trouvée")
        
        print()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        # Réinitialiser la simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Mettre le robot en position home
        if self.robot_joint_indices:
            for i, qpos_idx in enumerate(self.robot_joint_indices):
                if i < len(self.robot_info["home_qpos"]):
                    self.data.qpos[qpos_idx] = self.robot_info["home_qpos"][i]
        
        # Positionner le cube à sa position initiale
        self.initial_object_pos = self.task_config["object_initial_pos"].copy()
        
        if self.object_qpos_start is not None:
            self.data.qpos[self.object_qpos_start:self.object_qpos_start+3] = self.initial_object_pos
            # Quaternion identité pour l'orientation
            self.data.qpos[self.object_qpos_start+3:self.object_qpos_start+7] = [1, 0, 0, 0]
        
        # Position cible (depuis le site)
        self.target_pos = self.task_config["target_pos"].copy()
        
        self.current_step = 0
        self.object_grasped = False
        
        # Avancer la simulation pour stabiliser
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info
    
    def _get_obs(self) -> np.ndarray:
        """Construit le vecteur d'observation simplifié"""
        
        # Positions des joints du robot (6 joints)
        if self.robot_joint_indices:
            qpos_robot = np.array([self.data.qpos[i] for i in self.robot_joint_indices])
            qvel_robot = np.array([self.data.qvel[i] for i in self.robot_joint_indices])
        else:
            qpos_robot = self.data.qpos[:6].copy()
            qvel_robot = self.data.qvel[:6].copy()
        
        # Position du cube
        if self.object_qpos_start is not None:
            cube_pos = self.data.qpos[self.object_qpos_start:self.object_qpos_start+3].copy()
        else:
            cube_pos = self.initial_object_pos
        
        # Position cible (du site)
        if self.target_site_id is not None:
            goal_pos = self.data.site_xpos[self.target_site_id].copy()
        else:
            goal_pos = self.target_pos.copy()
        
        # Position end-effector
        ee_pos = self._get_ee_pos()
        
        obs = np.concatenate([qpos_robot, qvel_robot, cube_pos, goal_pos, ee_pos])
        
        return obs.astype(np.float32)
    
    def _get_ee_pos(self) -> np.ndarray:
        """Obtient la position du end-effector"""
        if self.ee_body_id is not None:
            return self.data.xpos[self.ee_body_id].copy()
        else:
            return np.zeros(3)
    
    def _get_object_pos(self) -> np.ndarray:
        """Obtient la position du cube"""
        if self.object_body_id is not None:
            return self.data.xpos[self.object_body_id].copy()
        elif self.object_qpos_start is not None:
            return self.data.qpos[self.object_qpos_start:self.object_qpos_start+3].copy()
        else:
            return self.initial_object_pos
    
    def _get_target_pos(self) -> np.ndarray:
        """Obtient la position de la cible"""
        if self.target_site_id is not None:
            return self.data.site_xpos[self.target_site_id].copy()
        else:
            return self.target_pos
    
    def _get_info(self) -> Dict:
        """Informations supplémentaires"""
        ee_pos = self._get_ee_pos()
        obj_pos = self._get_object_pos()
        target_pos = self._get_target_pos()
        
        dist_to_object = np.linalg.norm(ee_pos - obj_pos)
        dist_to_target = np.linalg.norm(obj_pos - target_pos)
        
        return {
            "distance_to_object": dist_to_object,
            "distance_to_target": dist_to_target,
            "object_grasped": self.object_grasped,
            "ee_position": ee_pos,
            "object_position": obj_pos,
            "target_position": target_pos,
            "grasp_success": False,
            "placement_success": False,
        }
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Exécute une action dans l'environnement"""
        self.current_step += 1
        
        # Appliquer l'action — clip aux vraies limites articulaires
        action = np.clip(action, JOINT_LIMITS["low"], JOINT_LIMITS["high"])
        self.data.ctrl[:self.model.nu] = action[:self.model.nu]
        
        # Avancer la simulation
        mujoco.mj_step(self.model, self.data)
        
        # Calculer la récompense
        reward, info = self._compute_reward()
        
        # Vérifier si l'épisode est terminé
        terminated = info.get("success", False)
        truncated = self.current_step >= self.max_episode_steps
        
        obs = self._get_obs()
        
        return obs, reward, terminated, truncated, info
    
    def _compute_reward(self) -> Tuple[float, Dict]:
        """Calcule la récompense pour pick-and-place"""
        info = self._get_info()
        reward = 0.0
        
        ee_pos = info["ee_position"]
        obj_pos = info["object_position"]
        target_pos = info["target_position"]
        dist_to_obj = info["distance_to_object"]
        dist_to_target = info["distance_to_target"]
        
        # Phase 1: Approche de l'objet
        if not self.object_grasped:
            reward -= dist_to_obj * 3.0
            
            if dist_to_obj < self.task_config["grasp_threshold"]:
                self.object_grasped = True
                reward += 15.0
                info["grasp_success"] = True
        
        # Phase 2: Transport vers la cible
        else:
            reward -= dist_to_target * 5.0
            
            if dist_to_target < self.task_config["placement_threshold"]:
                reward += 30.0
                info["success"] = True
                info["placement_success"] = True
        
        # Pénalité pour mouvement excessif
        if self.robot_joint_indices:
            qvel_robot = np.array([self.data.qvel[i] for i in self.robot_joint_indices])
            reward -= 0.01 * np.sum(np.abs(qvel_robot))
        
        return reward, info
    
    def get_camera_image(self, camera_name: str = None, width: int = 256, height: int = 256) -> np.ndarray:
        """
        Obtient une image depuis la caméra track_cam
        """
        if self.camera_renderer is None:
            self.camera_renderer = mujoco.Renderer(self.model, height=height, width=width)
        
        # Utiliser la caméra track_cam
        if camera_name is None:
            camera_name = self.element_names["camera"]
        
        try:
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        except:
            cam_id = 0  # Caméra par défaut
        
        self.camera_renderer.update_scene(self.data, camera=cam_id)
        image = self.camera_renderer.render()
        
        return image
    
    def render(self):
        """Affiche la simulation"""
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            else:
                self.viewer.sync()
    
    def close(self):
        """Ferme l'environnement"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# ============================================================================
# WRAPPERS DES MODÈLES (identiques à la version précédente)
# ============================================================================

class ModelWrapper:
    """Classe de base pour les wrappers de modèles"""
    
    def __init__(self, name: str):
        self.name = name
    
    def get_action(self, obs: np.ndarray, image: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """Méthode à implémenter par les sous-classes"""
        raise NotImplementedError


class SmolVLAWrapper(ModelWrapper):
    """Wrapper pour SmolVLA"""
    
    def __init__(self, use_real_model: bool = False):
        super().__init__("SmolVLA")
        self.use_real = use_real_model
        
        if self.use_real:
            print(f"📥 Chargement de {self.name} (vrai modèle)...")
            try:
                from transformers import AutoProcessor, AutoModelForVision2Seq
                import torch
                
                model_name = "HuggingFaceTB/SmolVLM-Instruct"
                self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    device_map="auto"
                )
                self.instruction = "Pick up the red cube and place it on the green goal."
                print(f"✅ {self.name} chargé")
            except Exception as e:
                print(f"❌ Erreur lors du chargement de {self.name}: {e}")
                print("   → Utilisation de la version simulée")
                self.use_real = False
        else:
            print(f"🔧 {self.name} en mode simulé (politique heuristique)")
    
    def get_action(self, obs: np.ndarray, image: Optional[np.ndarray] = None, info: Optional[Dict] = None) -> np.ndarray:
        """Obtient une action de SmolVLA"""
        
        if self.use_real and image is not None:
            return self._real_model_inference(image, obs)
        else:
            return self._simulated_policy(obs, info)
    
    def _real_model_inference(self, image: np.ndarray, obs: np.ndarray) -> np.ndarray:
        """Inférence avec le vrai modèle SmolVLA — génère du texte puis retombe sur la politique heuristique (SmolVLM-Instruct n'est pas un policy model)"""
        import torch
        
        pil_image = Image.fromarray(image.astype(np.uint8))
        
        inputs = self.processor(
            images=pil_image,
            text=self.instruction,
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=50, do_sample=False)
        
        generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
        print(f"SmolVLA output: {generated_text}")
        
        return self._simulated_policy(obs, None)
    
    def _simulated_policy(self, obs: np.ndarray, info: Optional[Dict]) -> np.ndarray:
        """Politique simulée pour SO100 (6 DoF)"""
        if info is None:
            info = {}
        
        # obs = [qpos_robot (6), qvel_robot (6), cube_pos (3), goal_pos (3), ee_pos (3)]
        cube_pos = obs[12:15]
        goal_pos = obs[15:18]
        ee_pos = obs[18:21]
        
        # Calculer la direction
        if not info.get("object_grasped", False):
            direction = cube_pos - ee_pos
            gripper_cmd = -1.0  # Ouvrir
        else:
            direction = goal_pos - ee_pos
            gripper_cmd = 1.0   # Fermer
        
        # Normaliser
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        
        # Créer l'action (6 actuateurs)
        # On ne contrôle que les 3 premiers joints + gripper de manière simplifiée
        action = np.zeros(6)
        action[0] = direction[1] * 0.3  # Rotation (axe Y -> contrôle X,Y)
        action[1] = -direction[2] * 0.4  # Pitch (axe Z)
        action[2] = direction[0] * 0.3  # Elbow
        action[5] = gripper_cmd  # Jaw
        
        return np.clip(action, -1, 1)


class OctoWrapper(ModelWrapper):
    """Wrapper pour Octo"""
    
    def __init__(self, use_real_model: bool = False, env=None):
        super().__init__("Octo")
        self.use_real = use_real_model
        self._step_counter = 0
        self.env = env  # référence à SO100PickPlaceEnv pour la couche IK
        
        if self.use_real:
            print(f"📥 Chargement de {self.name} (vrai modèle)...")
            try:
                from octo.model.octo_model import OctoModel
                
                self.model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small")
                print(f"✅ {self.name} chargé")
            except Exception as e:
                print(f"❌ Erreur lors du chargement de {self.name}: {e}")
                print("   → Utilisation de la version simulée")
                self.use_real = False
        else:
            print(f"🔧 {self.name} en mode simulé (politique heuristique)")
    
    def get_action(self, obs: np.ndarray, image: Optional[np.ndarray] = None, info: Optional[Dict] = None) -> np.ndarray:
        """Obtient une action d'Octo"""
        
        if self.use_real and image is not None:
            return self._real_model_inference(image, obs)
        else:
            return self._simulated_policy(obs, info)
    
    def _real_model_inference(self, image: np.ndarray, obs: np.ndarray) -> np.ndarray:
        """Inférence avec le vrai modèle Octo"""
        import jax

        pil_img = Image.fromarray(image.astype(np.uint8))
        pil_img = pil_img.resize((256, 256))
        image_resized = np.array(pil_img)

        # octo-small: (batch=1, window=1, H, W, C) — pas de proprio
        observation = {
            "image_primary": image_resized[np.newaxis, np.newaxis],
            "timestep_pad_mask": np.array([[True]]),
        }

        task = self.model.create_tasks(texts=["pick up the cube"])

        # RNG varié à chaque step pour éviter d'obtenir la même action en boucle
        self._step_counter += 1
        action = self.model.sample_actions(
            observation,
            task,
            unnormalization_statistics=self.model.dataset_statistics["bridge_dataset"]["action"],
            rng=jax.random.PRNGKey(self._step_counter),
        )
        # action: (batch=1, action_horizon, action_dim=7) — Bridge = [dx,dy,dz,droll,dpitch,dyaw,gripper]
        raw = np.array(action[0, 0])  # shape (7,)
        cart_delta = raw[:6].astype(np.float64)   # [dx,dy,dz,droll,dpitch,dyaw] en mètres/rad
        gripper    = float(raw[6]) if len(raw) > 6 else 0.0

        if self.env is not None and self.env.ee_body_id is not None:
            joint_delta = self._cartesian_to_joint_delta(cart_delta)
            # ctrl actuel = position cible courante des actuateurs (en rad)
            # On AJOUTE le delta pour un contrôle incrémental
            current_ctrl = self.env.data.ctrl[:6].copy().astype(np.float32)
            out = current_ctrl + joint_delta
            out[5] = float(np.clip(current_ctrl[5] + np.clip(gripper * 0.05, -0.1, 0.1),
                                   JOINT_LIMITS["low"][5], JOINT_LIMITS["high"][5]))
            # Clipper aux vraies limites articulaires
            out = np.clip(out, JOINT_LIMITS["low"], JOINT_LIMITS["high"])
        else:
            out = np.zeros(6, dtype=np.float32)
            out[:min(6, len(raw))] = raw[:6]

        return np.clip(out, JOINT_LIMITS["low"], JOINT_LIMITS["high"]).astype(np.float32)

    def _cartesian_to_joint_delta(self, cart_delta: np.ndarray) -> np.ndarray:
        """Cinématique inverse différentielle via pseudo-inverse amortie du Jacobien.

        Convertit un twist Cartésien [dx,dy,dz,droll,dpitch,dyaw] (espace opérationnel,
        sortie d'Octo-Bridge) en delta articulaire pour les 6 joints du SO100.

        Méthode : Damped Least Squares (DLS)  dq = J^T (J J^T + λ²I)^-1 · twist
        """
        model = self.env.model
        data  = self.env.data

        # Jacobiens MuJoCo : positions (3×nv) et rotations (3×nv)
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, self.env.ee_body_id)

        # Indices DOF des 6 joints du bras
        arm_dof = []
        for name in ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            arm_dof.append(int(model.jnt_dofadr[jid]))

        # Jacobien complet 6×6 (pos + rot) sur les joints du bras
        J = np.vstack([jacp[:, arm_dof], jacr[:, arm_dof]])

        # DLS : plus stable que pinv pur aux singularités
        damping = 0.05
        J_dls = J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(6))  # shape 6×6

        dq = J_dls @ cart_delta  # delta articulaire en rad

        # SCALE : pas d'amplification excessive, on veut ~0.02-0.05 rad/step
        SCALE = 2.0
        return (dq * SCALE).astype(np.float32)
    
    def _simulated_policy(self, obs: np.ndarray, info: Optional[Dict]) -> np.ndarray:
        """Politique simulée avec un peu de bruit"""
        if info is None:
            info = {}
        
        cube_pos = obs[12:15]
        goal_pos = obs[15:18]
        ee_pos = obs[18:21]
        
        if not info.get("object_grasped", False):
            direction = cube_pos - ee_pos
            speed = 0.35
            gripper_cmd = -0.8
        else:
            direction = goal_pos - ee_pos
            speed = 0.45
            gripper_cmd = 1.0
        
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        
        action = np.zeros(6)
        action[0] = direction[1] * speed
        action[1] = -direction[2] * speed
        action[2] = direction[0] * speed
        action[5] = gripper_cmd
        
        # Ajouter du bruit
        action += np.random.normal(0, 0.03, size=6)
        
        return np.clip(action, -1, 1)


class SACWrapper(ModelWrapper):
    """Wrapper pour SAC — utilise ReachCubeEnv + VecNormalize (environnement natif d'entraînement)."""

    def __init__(
        self,
        use_real_model: bool = False,
        checkpoint_path: Optional[str] = None,
        vecnorm_path: Optional[str] = None,
    ):
        super().__init__("SAC")
        self.use_real = use_real_model and checkpoint_path is not None
        self._native_vec_env = None
        self.model = None

        if self.use_real:
            print(f"📥 Chargement de {self.name} depuis {checkpoint_path}...")
            try:
                from stable_baselines3 import SAC
                from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
                from reach_cube_env import ReachCubeEnv

                self.model = SAC.load(checkpoint_path)

                # Recréer l'environnement natif sur lequel SAC a été entraîné
                raw_env = ReachCubeEnv(
                    task="reach",
                    max_episode_steps=500,
                    frame_skip=8,
                    random_cube=True,
                )
                self._native_vec_env = DummyVecEnv([lambda: raw_env])

                # Charger VecNormalize — INDISPENSABLE : sans ça les obs ne ressemblent
                # à rien de ce que le modèle a vu pendant l'entraînement
                vn_path = vecnorm_path or ""
                if vn_path and Path(vn_path).exists():
                    self._native_vec_env = VecNormalize.load(vn_path, self._native_vec_env)
                    self._native_vec_env.training    = False
                    self._native_vec_env.norm_reward = False
                    print(f"  ✅ VecNormalize chargé depuis {vn_path}")
                else:
                    print(f"  ⚠️  VecNormalize non trouvé ({vn_path}) — les performances seront dégradées")

                print(f"✅ {self.name} chargé (env natif: ReachCubeEnv + reach)")

            except Exception as e:
                print(f"❌ Erreur lors du chargement de {self.name}: {e}")
                print("   → Actions aléatoires")
                self.use_real = False
        else:
            print(f"🎲 {self.name} non entraîné (actions aléatoires - baseline)")

    # ------------------------------------------------------------------
    # Évaluation native (boucle indépendante avec le bon env)
    # ------------------------------------------------------------------

    def run_native_episodes(self, num_episodes: int, config: Dict) -> List[EpisodeMetrics]:
        """Évalue SAC sur son environnement natif (ReachCubeEnv + VecNormalize)."""
        print(f"\n{'='*70}")
        print(f"🤖 Test de {self.name}  (env natif: ReachCubeEnv, tâche: reach)")
        print(f"{'='*70}")

        results = []
        for ep in range(num_episodes):
            obs = self._native_vec_env.reset()
            done = False
            total_reward = 0.0
            steps = 0
            ep_success = False
            start_time = time.time()

            if config.get("verbose"):
                print(f"\n  Episode {ep+1}/{num_episodes}...")

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done_arr, info_list = self._native_vec_env.step(action)
                done = bool(done_arr[0])
                total_reward += float(reward[0])
                steps += 1
                if info_list:
                    ep_success = ep_success or bool(info_list[0].get("success", False))

            exec_time = time.time() - start_time
            metrics = EpisodeMetrics(
                success=ep_success,
                steps=steps,
                total_reward=total_reward,
                grasp_success=False,
                placement_success=False,
                execution_time=exec_time,
                model_name=self.name,
                task_name="reach",
                episode_id=ep,
            )
            results.append(metrics)

            if config.get("verbose"):
                print(
                    f"    → Succès: {metrics.success}, "
                    f"Récompense: {total_reward:.2f}, "
                    f"Steps: {steps}, "
                    f"Temps: {exec_time:.2f}s"
                )

        return results

    # ------------------------------------------------------------------
    # Fallback (utilisé seulement si _native_vec_env non disponible)
    # ------------------------------------------------------------------

    def get_action(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        if self.use_real and self.model is not None:
            action, _ = self.model.predict(obs, deterministic=True)
            return action
        return np.random.uniform(-1, 1, size=6)


# ============================================================================
# TESTEUR ZERO-SHOT
# ============================================================================

class ZeroShotTester:
    """Classe pour tester les modèles en zero-shot"""
    
    def __init__(self, env: SO100PickPlaceEnv, config: Dict):
        self.env = env
        self.config = config
        self.results = []
    
    def test_model(
        self,
        model_wrapper: ModelWrapper,
        num_episodes: int,
        use_vision: bool = False,
    ) -> List[EpisodeMetrics]:
        """Teste un modèle sur plusieurs épisodes.

        Si le wrapper expose ``run_native_episodes()`` (ex: SACWrapper avec son
        environnement natif chargé), la boucle lui est déléguée directement afin
        d'utiliser le bon espace d'observation / d'action et la VecNormalize.
        """
        # -- Délégation à l'environnement natif du modèle (SAC) --
        if isinstance(model_wrapper, SACWrapper) and model_wrapper._native_vec_env is not None:
            episode_results = model_wrapper.run_native_episodes(num_episodes, self.config)
            self.results.extend(episode_results)
            self._print_model_summary(model_wrapper.name, episode_results)
            return episode_results

        # -- Boucle générique pour les autres modèles (SmolVLA, Octo, ...) --
        print(f"\n{'='*70}")
        print(f"🤖 Test de {model_wrapper.name}")
        print(f"{'='*70}")

        episode_results = []

        for ep in range(num_episodes):
            obs, info = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            start_time = time.time()
            
            if self.config["verbose"]:
                print(f"\n  Episode {ep+1}/{num_episodes}...")
            
            while not done and steps < self.env.max_episode_steps:
                # Obtenir l'image seulement si le modèle l'utilise vraiment
                image = None
                if use_vision and getattr(model_wrapper, 'use_real', False):
                    image = self.env.get_camera_image()
                
                # Obtenir l'action du modèle
                action = model_wrapper.get_action(obs, image=image, info=info)

                # Afficher les actions prédites périodiquement
                log_every = self.config.get("log_actions_every", 0)
                if log_every and steps % log_every == 0:
                    joints = ["Rot", "Pitch", "Elbow", "WrPit", "WrRol", "Jaw"]
                    action_str = "  ".join(f"{n}:{v:+.3f}" for n, v in zip(joints, action))
                    print(f"    [step {steps:3d}] actions → {action_str}")

                # Exécuter l'action
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated
                
                # Afficher la simulation si demandé
                if self.config["render"]:
                    self.env.render()
                    time.sleep(0.01)
            
            exec_time = time.time() - start_time
            
            # Créer les métriques
            metrics = EpisodeMetrics(
                success=info.get("success", False),
                steps=steps,
                total_reward=total_reward,
                grasp_success=info.get("grasp_success", False),
                placement_success=info.get("placement_success", False),
                execution_time=exec_time,
                model_name=model_wrapper.name,
                task_name="pick_and_place",
                episode_id=ep
            )
            episode_results.append(metrics)
            
            if self.config["verbose"]:
                print(f"    → Succès: {metrics.success}, "
                      f"Récompense: {total_reward:.2f}, "
                      f"Steps: {steps}, "
                      f"Temps: {exec_time:.2f}s")
        
        self.results.extend(episode_results)
        self._print_model_summary(model_wrapper.name, episode_results)
        
        return episode_results
    
    def _print_model_summary(self, model_name: str, episodes: List[EpisodeMetrics]):
        """Affiche un résumé des performances d'un modèle"""
        successes = sum(1 for e in episodes if e.success)
        grasps = sum(1 for e in episodes if e.grasp_success)
        
        print(f"\n  📊 Résumé {model_name}:")
        print(f"    • Taux de succès: {successes}/{len(episodes)} ({successes/len(episodes)*100:.1f}%)")
        print(f"    • Taux de saisie: {grasps}/{len(episodes)} ({grasps/len(episodes)*100:.1f}%)")
        print(f"    • Récompense moyenne: {np.mean([e.total_reward for e in episodes]):.2f}")
        print(f"    • Steps moyens: {np.mean([e.steps for e in episodes]):.1f}")
    
    def save_results(self, output_path: str = "benchmark_results.json"):
        """Sauvegarde les résultats"""
        by_model = {}
        for metric in self.results:
            if metric.model_name not in by_model:
                by_model[metric.model_name] = []
            by_model[metric.model_name].append(asdict(metric))
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "configuration": self.config,
            "xml_path": self.env.xml_path,
            "total_episodes": len(self.results),
            "models": {}
        }
        
        for model_name, episodes in by_model.items():
            successes = sum(1 for e in episodes if e["success"])
            grasps = sum(1 for e in episodes if e["grasp_success"])
            
            report["models"][model_name] = {
                "episodes": episodes,
                "statistics": {
                    "success_rate": successes / len(episodes),
                    "grasp_success_rate": grasps / len(episodes),
                    "avg_reward": np.mean([e["total_reward"] for e in episodes]),
                    "avg_steps": np.mean([e["steps"] for e in episodes]),
                    "avg_time": np.mean([e["execution_time"] for e in episodes]),
                }
            }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✅ Résultats sauvegardés: {output_path}")
        
        return report


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """Fonction principale"""
    
    print("="*70)
    print("🚀 BENCHMARK ZERO-SHOT - SmolVLA vs Octo vs SAC")
    print("   Bras SO100 (Hugging Face) + MuJoCo")
    print("   Tâche: Pick-and-Place")
    print("="*70)
    
    # Vérifier que le fichier XML existe
    if not Path(SCENE_XML_PATH).exists():
        print(f"\n❌ ERREUR: Fichier XML non trouvé: {SCENE_XML_PATH}")
        print(f"   → Assurez-vous que scene_pick_place.xml et so_arm100.xml")
        print(f"      sont dans le même répertoire que ce script")
        print(f"   → Chemin actuel: {Path(SCENE_XML_PATH).absolute()}")
        return
    
    print(f"\n✅ Scène trouvée: {SCENE_XML_PATH}")
    
    # Créer l'environnement
    env = SO100PickPlaceEnv(
        xml_path=SCENE_XML_PATH,
        render_mode="human" if BENCHMARK_CONFIG["render"] else None,
        task_config=TASK_CONFIG,
        element_names=XML_ELEMENT_NAMES,
        robot_info=ROBOT_JOINTS
    )
    
    # Créer le testeur
    tester = ZeroShotTester(env, BENCHMARK_CONFIG)
    
    # Initialiser les modèles
    models = [
        SmolVLAWrapper(use_real_model=USE_REAL_MODELS["smolvla"]),
        OctoWrapper(use_real_model=USE_REAL_MODELS["octo"], env=env),
        SACWrapper(
            use_real_model=USE_REAL_MODELS["sac"],
            checkpoint_path=MODEL_CHECKPOINTS["sac"],
            vecnorm_path=MODEL_CHECKPOINTS.get("sac_vecnorm"),
        ),
    ]
    
    # Tester chaque modèle
    num_episodes = BENCHMARK_CONFIG["num_episodes"]
    
    for model in models:
        use_vision = isinstance(model, (SmolVLAWrapper, OctoWrapper))
        tester.test_model(model, num_episodes=num_episodes, use_vision=use_vision)
    
    # Sauvegarder les résultats
    report = tester.save_results("benchmark_results.json")
    
    # Afficher le résumé final
    print("\n" + "="*70)
    print("📊 RÉSUMÉ FINAL")
    print("="*70)
    
    for model_name, data in report["models"].items():
        stats = data["statistics"]
        print(f"\n{model_name}:")
        print(f"  Taux de succès: {stats['success_rate']*100:.1f}%")
        print(f"  Taux de saisie: {stats['grasp_success_rate']*100:.1f}%")
        print(f"  Récompense moyenne: {stats['avg_reward']:.2f}")
        print(f"  Steps moyens: {stats['avg_steps']:.1f}")

    
    env.close()


if __name__ == "__main__":
    main()