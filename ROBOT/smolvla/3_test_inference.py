"""
Script pour tester SmolVLA entraîné sur le robot SO100
Usage: python 3_test_inference.py --model-path ./checkpoints/smolvla_so100_20240227/final_model --task "pick the red cube"
"""

import argparse
import os
import sys
from pathlib import Path
import time
import torch
from PIL import Image
import numpy as np

try:
    from transformers import AutoProcessor, AutoModelForVision2Seq
except ImportError as e:
    print("❌ Erreur d'importation:")
    print(f"   {e}")
    print("\n💡 Assure-toi que l'environnement virtuel est activé:")
    print("   source venv_smolvla/bin/activate")
    sys.exit(1)

from config import SO100_CONFIG, CAMERA_CONFIG


class SmolVLAController:
    """Contrôleur pour utiliser SmolVLA avec le SO100"""
    
    def __init__(self, model_path, device="auto"):
        """
        Args:
            model_path: Chemin vers le modèle entraîné
            device: "cuda", "mps", "cpu", ou "auto" pour auto-détection
        """
        print("🤖 Initialisation du contrôleur SmolVLA...")
        
        # Déterminer le device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                print("✓ GPU détecté (CUDA)")
            elif torch.backends.mps.is_available():
                self.device = "mps"
                print("✓ GPU détecté (Apple Silicon MPS)")
            else:
                self.device = "cpu"
                print("⚠️  Utilisation du CPU (sera lent)")
        else:
            self.device = device
        
        # Charger le modèle
        print(f"📂 Chargement du modèle depuis: {model_path}")
        
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16
            ).to(self.device)
            
            self.model.eval()
            print("✓ Modèle chargé et prêt")
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle: {e}")
            raise
    
    def predict_action(self, image, task_instruction, robot_state=None):
        """
        Prédit l'action à partir d'une image et d'une instruction
        
        Args:
            image: PIL Image ou numpy array
            task_instruction: str, description de la tâche
            robot_state: optionnel, état actuel du robot
        
        Returns:
            action: numpy array [6] (positions des 6 moteurs)
        """
        # Convertir l'image si nécessaire
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Créer le prompt
        prompt = f"In: What action should the robot take to {task_instruction}?\nOut:"
        
        # Préparer les inputs
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Extraire l'action prédite
            # Note: L'extraction exacte dépend de l'architecture du modèle
            # Ici on suppose que les derniers 6 tokens correspondent aux actions
            predicted_action = outputs.last_hidden_state[0, -1, :6]
            
            # Convertir en numpy
            action = predicted_action.cpu().numpy()
        
        return action


def test_inference_simple(args):
    """Test simple avec des images de test"""
    
    print("\n" + "="*60)
    print("🧪 TEST D'INFÉRENCE SMOLVLA (Mode Simple)")
    print("="*60)
    
    # Vérifier que le modèle existe
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Modèle non trouvé: {model_path}")
        print("   Lance d'abord 2_train_smolvla.py pour entraîner un modèle")
        sys.exit(1)
    
    # Initialiser le contrôleur
    controller = SmolVLAController(str(model_path))
    
    # Créer une image de test (ou charger depuis fichier)
    print("\n📸 Création d'une image de test...")
    test_image = Image.new('RGB', (640, 480), color=(100, 150, 200))
    
    # Prédire l'action
    print(f"🎯 Tâche: {args.task}")
    print("🔮 Prédiction de l'action...")
    
    start_time = time.time()
    action = controller.predict_action(test_image, args.task)
    inference_time = time.time() - start_time
    
    print(f"✓ Action prédite en {inference_time*1000:.1f}ms")
    print(f"\n📊 Action:")
    motor_names = ["Shoulder Pan", "Shoulder Lift", "Elbow", "Wrist Flex", "Wrist Roll", "Gripper"]
    for i, (name, value) in enumerate(zip(motor_names, action)):
        print(f"   {name:15s}: {value:+7.3f}")
    
    print("\n💡 Pour tester sur le robot réel, utilise --robot")


def test_inference_robot(args):
    """Test avec le robot SO100 réel"""
    
    print("\n" + "="*60)
    print("🤖 TEST D'INFÉRENCE SMOLVLA (Mode Robot)")
    print("="*60)
    
    # Vérifier le modèle
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Modèle non trouvé: {model_path}")
        sys.exit(1)
    
    # Initialiser le contrôleur
    controller = SmolVLAController(str(model_path))
    
    # Importer et initialiser le robot
    try:
        from lerobot.common.robot_devices.robots.factory import make_robot
        
        print("\n🔌 Connexion au robot SO100...")
        robot = make_robot(
            robot_type="so100",
            port=SO100_CONFIG["port"],
            baudrate=SO100_CONFIG["baudrate"],
            motors=SO100_CONFIG["motors"],
            cameras=CAMERA_CONFIG
        )
        robot.connect()
        print("✓ Robot connecté")
        
        # Move to home
        print("🏠 Déplacement vers position initiale...")
        robot.move_to_home_position(SO100_CONFIG["home_position"])
        time.sleep(1)
        
    except Exception as e:
        print(f"❌ Erreur de connexion au robot: {e}")
        return
    
    print("\n" + "="*60)
    print("📋 CONTRÔLE AUTONOME")
    print("="*60)
    print(f"🎯 Tâche: {args.task}")
    print(f"⏱️  Durée: {args.duration}s")
    print("⚠️  Surveille le robot et appuie sur CTRL+C pour arrêter!")
    print("="*60)
    
    input("\n▶️  Appuie sur ENTRÉE pour démarrer...")
    
    # Boucle de contrôle
    try:
        start_time = time.time()
        step_count = 0
        
        print("\n🔴 EXÉCUTION EN COURS...\n")
        
        while time.time() - start_time < args.duration:
            # Capturer l'image
            observation = robot.get_observation()
            image = observation['images']['top']
            robot_state = observation['state']
            
            # Prédire l'action
            action = controller.predict_action(image, args.task, robot_state)
            
            # Envoyer l'action au robot
            robot.send_action(action)
            
            # Stats
            step_count += 1
            if step_count % 10 == 0:
                elapsed = time.time() - start_time
                fps = step_count / elapsed
                print(f"⏱️  {elapsed:.1f}s | {step_count} steps | {fps:.1f} Hz")
            
            # Contrôle de la fréquence
            time.sleep(1.0 / 30)  # 30 Hz
        
        print("\n✅ Exécution terminée")
        
    except KeyboardInterrupt:
        print("\n⏸️  Arrêt par l'utilisateur")
    
    finally:
        # Retour à la position home
        print("🏠 Retour à la position initiale...")
        robot.move_to_home_position(SO100_CONFIG["home_position"])
        robot.disconnect()
        print("✓ Robot déconnecté")
    
    print(f"\n📊 Statistiques:")
    print(f"   Steps exécutés: {step_count}")
    print(f"   Durée: {time.time() - start_time:.1f}s")
    print(f"   Fréquence moyenne: {step_count / (time.time() - start_time):.1f} Hz")


def visualize_predictions(args):
    """Visualise les prédictions sur des images"""
    
    print("\n" + "="*60)
    print("👁️  VISUALISATION DES PRÉDICTIONS")
    print("="*60)
    
    # Charger le modèle
    controller = SmolVLAController(args.model_path)
    
    # Importer matplotlib
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  matplotlib non installé, impossible de visualiser")
        print("   pip install matplotlib")
        return
    
    # Créer quelques images de test
    test_images = [
        Image.new('RGB', (640, 480), color=(200, 100, 100)),  # Rouge
        Image.new('RGB', (640, 480), color=(100, 200, 100)),  # Vert
        Image.new('RGB', (640, 480), color=(100, 100, 200)),  # Bleu
    ]
    
    fig, axes = plt.subplots(len(test_images), 2, figsize=(12, 4*len(test_images)))
    
    motor_names = ["Shoulder\nPan", "Shoulder\nLift", "Elbow", "Wrist\nFlex", "Wrist\nRoll", "Gripper"]
    
    for i, img in enumerate(test_images):
        # Prédire
        action = controller.predict_action(img, args.task)
        
        # Afficher l'image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Test Image {i+1}")
        axes[i, 0].axis('off')
        
        # Afficher les actions prédites
        axes[i, 1].bar(range(6), action)
        axes[i, 1].set_xticks(range(6))
        axes[i, 1].set_xticklabels(motor_names, rotation=45)
        axes[i, 1].set_ylabel("Action Value")
        axes[i, 1].set_title(f"Predicted Actions")
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("predictions_visualization.png", dpi=150, bbox_inches='tight')
    print("✓ Visualisation sauvegardée: predictions_visualization.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Tester l'inférence SmolVLA")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Chemin vers le modèle entraîné"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="pick the red cube",
        help="Instruction de la tâche"
    )
    parser.add_argument(
        "--robot",
        action="store_true",
        help="Tester sur le robot réel (sinon test simple)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Durée de l'exécution en secondes (mode robot)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualiser les prédictions"
    )
    
    args = parser.parse_args()
    
    try:
        if args.visualize:
            visualize_predictions(args)
        elif args.robot:
            test_inference_robot(args)
        else:
            test_inference_simple(args)
    except KeyboardInterrupt:
        print("\n\n👋 Test interrompu")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
