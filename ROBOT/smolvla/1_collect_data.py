"""
Script pour collecter des données de démonstration avec le SO100
Usage: python 1_collect_data.py --task "pick the red cube" --num-episodes 50
"""

import argparse
import os
import sys
from pathlib import Path
import time
from datetime import datetime
import json

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.robot_devices.robots.factory import make_robot
    from lerobot.scripts.control_robot import record
except ImportError as e:
    print("❌ Erreur d'importation LeRobot:")
    print(f"   {e}")
    print("\n💡 Assure-toi que l'environnement virtuel est activé:")
    print("   source venv_smolvla/bin/activate")
    sys.exit(1)

from config import SO100_CONFIG, CAMERA_CONFIG, DATASET_CONFIG, PATHS


def check_robot_connection(port):
    """Vérifie si le robot est connecté"""
    import serial.tools.list_ports
    
    ports = [p.device for p in serial.tools.list_ports.comports()]
    
    print(f"🔍 Ports série disponibles: {ports}")
    
    if port not in ports:
        print(f"⚠️  Port {port} non trouvé!")
        print("   Trouve ton port avec: ls /dev/tty.*")
        print("   Puis modifie SO100_CONFIG['port'] dans config.py")
        return False
    
    return True


def setup_directories():
    """Crée les répertoires nécessaires"""
    for path in PATHS.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    print("✓ Répertoires créés")


def collect_data(args):
    """Collecte des démonstrations pour une tâche"""
    
    print("\n" + "="*60)
    print("📹 COLLECTE DE DONNÉES POUR SO100")
    print("="*60)
    
    # Vérifier la connexion
    if not check_robot_connection(SO100_CONFIG["port"]):
        return
    
    # Setup
    setup_directories()
    
    # Nom du dataset
    task_name = args.task.replace(" ", "_").lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = f"{args.repo_id}/{task_name}_{timestamp}" if args.repo_id else f"local/{task_name}_{timestamp}"
    
    print(f"\n📦 Dataset: {dataset_name}")
    print(f"📝 Tâche: {args.task}")
    print(f"🎬 Nombre d'épisodes: {args.num_episodes}")
    print(f"⏱️  Durée par épisode: {DATASET_CONFIG['episode_length']}s")
    
    # Configuration du robot
    robot_config = {
        "robot_type": "so100",
        "port": SO100_CONFIG["port"],
        "baudrate": SO100_CONFIG["baudrate"],
        "motors": SO100_CONFIG["motors"],
    }
    
    # Configuration des caméras
    camera_config = CAMERA_CONFIG
    
    print("\n⚙️  Configuration:")
    print(f"   Port: {robot_config['port']}")
    print(f"   Caméras: {list(camera_config.keys())}")
    
    # Initialiser le robot
    try:
        print("\n🤖 Connexion au robot...")
        robot = make_robot(**robot_config, cameras=camera_config)
        robot.connect()
        print("✓ Robot connecté")
        
        # Move to home position
        print("🏠 Déplacement vers position initiale...")
        robot.move_to_home_position(SO100_CONFIG["home_position"])
        time.sleep(1)
        
    except Exception as e:
        print(f"❌ Erreur de connexion au robot: {e}")
        print("\n💡 Vérifications:")
        print("   1. Le robot est-il branché et allumé?")
        print("   2. Le port est-il correct dans config.py?")
        print("   3. As-tu les permissions? (sudo si nécessaire)")
        return
    
    print("\n" + "="*60)
    print("📋 INSTRUCTIONS")
    print("="*60)
    print("1. À chaque épisode, le robot retournera à sa position initiale")
    print("2. Attends le signal de démarrage (3 secondes)")
    print("3. Téléopère le robot pour accomplir la tâche")
    print("4. L'enregistrement s'arrête automatiquement après 30s")
    print("5. Prends le temps de reset la scène entre chaque épisode")
    print("\n💡 Conseils:")
    print("   - Varie légèrement les positions des objets")
    print("   - Montre différentes approches de la tâche")
    print("   - Sois fluide dans tes mouvements")
    print("="*60)
    
    input("\n▶️  Appuie sur ENTRÉE pour commencer...")
    
    # Collecter les épisodes
    episodes_data = []
    
    for episode_idx in range(args.num_episodes):
        print(f"\n{'='*60}")
        print(f"🎬 ÉPISODE {episode_idx + 1}/{args.num_episodes}")
        print(f"{'='*60}")
        
        # Reset à la position home
        print("🏠 Reset à la position initiale...")
        robot.move_to_home_position(SO100_CONFIG["home_position"])
        
        print(f"\n⏸️  Prépare la scène (tu as {DATASET_CONFIG['reset_time']}s)")
        time.sleep(DATASET_CONFIG["reset_time"])
        
        print(f"⏳ Démarrage dans {DATASET_CONFIG['warmup_time']}s...")
        time.sleep(DATASET_CONFIG['warmup_time'])
        
        print("🔴 ENREGISTREMENT EN COURS...")
        
        # Enregistrer l'épisode
        try:
            episode_data = []
            start_time = time.time()
            
            while time.time() - start_time < DATASET_CONFIG['episode_length']:
                # Capturer observation
                obs = robot.get_observation()
                
                # Capturer action (position actuelle comme proxy)
                action = robot.get_current_action()
                
                # Stocker
                episode_data.append({
                    'observation': obs,
                    'action': action,
                    'task': args.task,
                    'timestamp': time.time() - start_time
                })
                
                # Contrôle à la fréquence du dataset
                time.sleep(1.0 / DATASET_CONFIG['fps'])
            
            print("✓ Épisode enregistré")
            episodes_data.append(episode_data)
            
            # Sauvegarder progressivement
            save_episode(episode_data, dataset_name, episode_idx)
            
        except KeyboardInterrupt:
            print("\n⏸️  Interruption par l'utilisateur")
            if input("Continuer? (o/n): ").lower() != 'o':
                break
        except Exception as e:
            print(f"❌ Erreur lors de l'enregistrement: {e}")
            if input("Réessayer cet épisode? (o/n): ").lower() == 'o':
                continue
            else:
                break
    
    # Déconnexion
    robot.disconnect()
    print("\n✅ Collecte terminée!")
    print(f"📦 {len(episodes_data)} épisodes enregistrés")
    print(f"💾 Données sauvegardées dans: {PATHS['data_dir']}/{dataset_name}")
    
    # Sauvegarder les métadonnées
    save_metadata(dataset_name, args.task, len(episodes_data))


def save_episode(episode_data, dataset_name, episode_idx):
    """Sauvegarde un épisode"""
    episode_dir = Path(PATHS['data_dir']) / dataset_name / f"episode_{episode_idx:04d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder les données (format simplifié pour l'exemple)
    with open(episode_dir / "data.json", "w") as f:
        # Convertir les observations en format sérialisable
        serializable_data = []
        for step in episode_data:
            serializable_data.append({
                'task': step['task'],
                'timestamp': step['timestamp'],
                'action': step['action'].tolist() if hasattr(step['action'], 'tolist') else step['action'],
                # Les images seraient sauvegardées séparément dans un vrai setup
            })
        json.dump(serializable_data, f, indent=2)


def save_metadata(dataset_name, task, num_episodes):
    """Sauvegarde les métadonnées du dataset"""
    metadata = {
        'dataset_name': dataset_name,
        'task': task,
        'num_episodes': num_episodes,
        'fps': DATASET_CONFIG['fps'],
        'episode_length': DATASET_CONFIG['episode_length'],
        'robot': 'so100',
        'created_at': datetime.now().isoformat(),
    }
    
    metadata_path = Path(PATHS['data_dir']) / dataset_name / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Collecter des données avec SO100")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Description de la tâche (ex: 'pick the red cube')"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=50,
        help="Nombre d'épisodes à enregistrer (défaut: 50)"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="HuggingFace repo ID (ex: 'ton-username/dataset-name'). Si non spécifié, sauvegarde localement."
    )
    
    args = parser.parse_args()
    
    try:
        collect_data(args)
    except KeyboardInterrupt:
        print("\n\n👋 Collecte annulée par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
