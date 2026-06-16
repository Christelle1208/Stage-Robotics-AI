"""
Script pour entraîner SmolVLA sur les données collectées
Usage: python 2_train_smolvla.py --dataset-path ./data/local/pick_cube_20240227
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import json
from datetime import datetime

try:
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from transformers import TrainingArguments, Trainer
    from torch.utils.data import Dataset, DataLoader
    import wandb
except ImportError as e:
    print("❌ Erreur d'importation:")
    print(f"   {e}")
    print("\n💡 Assure-toi que l'environnement virtuel est activé:")
    print("   source venv_smolvla/bin/activate")
    sys.exit(1)

from config import TRAINING_CONFIG, SMOLVLA_CONFIG, PATHS


class SO100Dataset(Dataset):
    """Dataset pour SmolVLA avec données SO100"""
    
    def __init__(self, dataset_path, processor):
        self.dataset_path = Path(dataset_path)
        self.processor = processor
        
        # Charger les métadonnées
        with open(self.dataset_path / "metadata.json", "r") as f:
            self.metadata = json.load(f)
        
        # Lister tous les épisodes
        self.episodes = sorted(list(self.dataset_path.glob("episode_*")))
        
        print(f"📊 Dataset chargé: {len(self.episodes)} épisodes")
        print(f"   Tâche: {self.metadata['task']}")
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        episode_dir = self.episodes[idx]
        
        # Charger les données de l'épisode
        with open(episode_dir / "data.json", "r") as f:
            episode_data = json.load(f)
        
        # Pour simplifier, on prend un frame aléatoire
        # Dans un vrai setup, on itérerait sur tous les frames
        import random
        frame = random.choice(episode_data)
        
        # Créer l'input pour SmolVLA
        # Note: Dans ce script simplifié, on suppose que les images sont sauvegardées
        # Dans un vrai setup LeRobot, elles seraient chargées depuis le dataset
        
        task_instruction = frame['task']
        # image = load_image(episode_dir / f"frame_{frame['timestamp']}.jpg")
        # Pour cet exemple, on crée une image dummy
        from PIL import Image
        import numpy as np
        image = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        
        action = torch.tensor(frame['action'], dtype=torch.float32)
        
        # Préparer les inputs
        inputs = self.processor(
            text=f"In: What action should the robot take to {task_instruction}?\nOut:",
            images=image,
            return_tensors="pt"
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'labels': action
        }


def setup_training(args):
    """Configure l'entraînement"""
    
    print("\n" + "="*60)
    print("🧠 ENTRAÎNEMENT SMOLVLA POUR SO100")
    print("="*60)
    
    # Vérifier le dataset
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"❌ Dataset non trouvé: {dataset_path}")
        print("   Lance d'abord 1_collect_data.py pour collecter des données")
        sys.exit(1)
    
    print(f"\n📦 Dataset: {dataset_path}")
    
    # Device
    if torch.cuda.is_available():
        device = "cuda"
        print("✓ GPU détecté (CUDA)")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("✓ GPU détecté (Apple Silicon MPS)")
    else:
        device = "cpu"
        print("⚠️  Pas de GPU, utilisation du CPU (sera lent!)")
    
    # Charger le modèle pré-entraîné
    print(f"\n🤖 Chargement du modèle: {SMOLVLA_CONFIG['pretrained_model']}")
    print("   (Cela peut prendre quelques minutes la première fois...)")
    
    try:
        processor = AutoProcessor.from_pretrained(
            SMOLVLA_CONFIG['pretrained_model'],
            trust_remote_code=True
        )
        
        model = AutoModelForVision2Seq.from_pretrained(
            SMOLVLA_CONFIG['pretrained_model'],
            trust_remote_code=True,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16
        ).to(device)
        
        print("✓ Modèle chargé")
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        print("\n💡 Assure-toi d'avoir une connexion internet pour télécharger le modèle")
        sys.exit(1)
    
    # Configurer LoRA si activé
    if SMOLVLA_CONFIG['use_lora']:
        print(f"\n🔧 Configuration LoRA (rank={SMOLVLA_CONFIG['lora_rank']})")
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=SMOLVLA_CONFIG['lora_rank'],
            lora_alpha=SMOLVLA_CONFIG['lora_alpha'],
            target_modules=["q_proj", "v_proj"],  # Adapter selon le modèle
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Créer le dataset
    print("\n📊 Chargement du dataset...")
    train_dataset = SO100Dataset(dataset_path, processor)
    
    # Configuration de l'entraînement
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(PATHS['checkpoints_dir']) / f"smolvla_so100_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=TRAINING_CONFIG['num_epochs'],
        per_device_train_batch_size=TRAINING_CONFIG['batch_size'],
        gradient_accumulation_steps=TRAINING_CONFIG['gradient_accumulation_steps'],
        learning_rate=TRAINING_CONFIG['learning_rate'],
        warmup_steps=TRAINING_CONFIG['warmup_steps'],
        logging_steps=100,
        save_steps=TRAINING_CONFIG['save_freq'],
        eval_steps=TRAINING_CONFIG['eval_freq'],
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to="wandb" if args.use_wandb else "none",
        fp16=device == "cuda",  # Mixed precision sur CUDA
        dataloader_num_workers=0,  # Pas de multiprocessing sur Mac
    )
    
    print(f"\n⚙️  Configuration:")
    print(f"   Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"   Learning rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"   Epochs: {TRAINING_CONFIG['num_epochs']}")
    print(f"   Device: {device}")
    print(f"   Output: {output_dir}")
    
    # Initialiser Weights & Biases si demandé
    if args.use_wandb:
        print("\n📊 Initialisation de Weights & Biases...")
        wandb.init(
            project="smolvla-so100",
            name=f"training_{timestamp}",
            config={
                **TRAINING_CONFIG,
                **SMOLVLA_CONFIG,
                "dataset": str(dataset_path)
            }
        )
    
    # Custom Trainer pour les actions
    class VLATrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            # Forward pass
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                pixel_values=inputs['pixel_values']
            )
            
            # Extraire les prédictions d'action
            # (L'architecture exacte dépend du modèle)
            predicted_actions = outputs.last_hidden_state[:, -1, :SMOLVLA_CONFIG['action_dim']]
            
            # MSE loss sur les actions
            loss = torch.nn.functional.mse_loss(
                predicted_actions,
                inputs['labels']
            )
            
            return (loss, outputs) if return_outputs else loss
    
    # Créer le trainer
    trainer = VLATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=lambda data: {
            'input_ids': torch.stack([f['input_ids'] for f in data]),
            'attention_mask': torch.stack([f['attention_mask'] for f in data]),
            'pixel_values': torch.stack([f['pixel_values'] for f in data]),
            'labels': torch.stack([f['labels'] for f in data]),
        }
    )
    
    return trainer, model, processor, output_dir


def train(args):
    """Lance l'entraînement"""
    
    # Setup
    trainer, model, processor, output_dir = setup_training(args)
    
    # Entraînement
    print("\n" + "="*60)
    print("🚀 DÉMARRAGE DE L'ENTRAÎNEMENT")
    print("="*60)
    print("\n⏳ Cela peut prendre plusieurs heures...")
    print("💡 Surveille les logs dans le terminal ou sur Weights & Biases\n")
    
    try:
        trainer.train()
        
        print("\n✅ Entraînement terminé!")
        
        # Sauvegarder le modèle final
        final_model_path = output_dir / "final_model"
        trainer.save_model(str(final_model_path))
        processor.save_pretrained(str(final_model_path))
        
        print(f"💾 Modèle sauvegardé dans: {final_model_path}")
        
        # Résumé
        print("\n" + "="*60)
        print("📊 RÉSUMÉ")
        print("="*60)
        print(f"Checkpoints: {output_dir}")
        print(f"Modèle final: {final_model_path}")
        print("\nProchaine étape: Lance 3_test_inference.py pour tester le modèle")
        
    except KeyboardInterrupt:
        print("\n⏸️  Entraînement interrompu par l'utilisateur")
        print(f"💾 Checkpoints sauvegardés dans: {output_dir}")
    except Exception as e:
        print(f"\n❌ Erreur pendant l'entraînement: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Entraîner SmolVLA sur données SO100")
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Chemin vers le dataset collecté"
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Utiliser Weights & Biases pour le tracking"
    )
    
    args = parser.parse_args()
    
    try:
        train(args)
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
