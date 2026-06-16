"""
Configuration pour le robot SO100
Adapter les paramètres selon ton setup
"""

# Configuration du port série (à adapter selon ton Mac)
# Pour trouver le port: ls /dev/tty.* dans le terminal

LEADER_PORT   = "/dev/tty.usbmodem58FA0928531"   # bras que vous bougez
FOLLOWER_PORT = "/dev/tty.usbmodem59700734041"   # bras qui suit

SO100_CONFIG = {
    "port": LEADER_PORT,
    "baudrate": 1000000,
    "motors": {
        "shoulder_pan": 1,
        "shoulder_lift": 2,
        "elbow_flex": 3,
        "wrist_flex": 4,
        "wrist_roll": 5,
        "gripper": 6,
    },
    "motor_limits": {
        # Limites en degrés pour chaque moteur (sécurité)
        1: (-180, 180),
        2: (-90, 90),
        3: (-120, 120),
        4: (-90, 90),
        5: (-180, 180),
        6: (0, 100),  # Gripper: 0=fermé, 100=ouvert
    },
    "home_position": [0, -45, 90, 45, 0, 50],  # Position de repos
}

# Configuration des caméras
CAMERA_CONFIG = {
    "top": {
        "index": 0,  # Index de la caméra (0 pour webcam par défaut)
        "width": 640,
        "height": 480,
        "fps": 30,
    },
    # Ajouter d'autres caméras si nécessaire
    # "wrist": {
    #     "index": 1,
    #     "width": 640,
    #     "height": 480,
    #     "fps": 30,
    # }
}

# Configuration de l'entraînement
TRAINING_CONFIG = {
    "batch_size": 8,
    "num_epochs": 100,
    "learning_rate": 1e-5,
    "warmup_steps": 500,
    "eval_freq": 1000,  # Évaluer tous les N steps
    "save_freq": 5000,  # Sauvegarder tous les N steps
    "gradient_accumulation_steps": 4,
    "max_grad_norm": 1.0,
}

# Configuration du dataset
DATASET_CONFIG = {
    "fps": 30,
    "episode_length": 30,  # secondes
    "num_episodes": 50,
    "warmup_time": 3,  # secondes avant l'enregistrement
    "reset_time": 10,  # temps pour reset entre épisodes
}

# Configuration SmolVLA
SMOLVLA_CONFIG = {
    "pretrained_model": "HuggingFaceTB/SmolVLM-Instruct",
    "use_lora": True,
    "lora_rank": 16,
    "lora_alpha": 32,
    "freeze_vision_encoder": False,
    "action_dim": 6,  # SO100 a 6 DoF
    "normalize_actions": True,
}

# Tâches d'exemple
EXAMPLE_TASKS = [
    "pick the red cube",
    "grasp the blue bottle",
    "push the green block to the right",
    "place the object in the box",
    "open the drawer",
]

# Chemins
PATHS = {
    "data_dir": "./data",
    "checkpoints_dir": "./checkpoints",
    "logs_dir": "./logs",
    "videos_dir": "./videos",
}
