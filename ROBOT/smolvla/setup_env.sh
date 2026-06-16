#!/bin/bash

# Script de setup pour SmolVLA + LeRobot sur Mac
# Testé sur macOS avec Python 3.10+

set -e  # Arrêter en cas d'erreur

echo "🚀 Installation de l'environnement SmolVLA pour SO100"
echo "=================================================="

# Vérifier la version de Python
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Version Python détectée: $python_version"

# Créer l'environnement virtuel
echo ""
echo "📦 Création de l'environnement virtuel..."
python3 -m venv venv_smolvla

# Activer l'environnement
echo "✓ Activation de l'environnement virtuel"
source venv_smolvla/bin/activate

# Mettre à jour pip
echo ""
echo "⬆️  Mise à jour de pip..."
pip install --upgrade pip setuptools wheel

# Installer PyTorch (version CPU pour Mac M1/M2/M3 ou Intel)
echo ""
echo "🔥 Installation de PyTorch..."
if [[ $(uname -m) == 'arm64' ]]; then
    echo "  → Détection Mac Apple Silicon (M1/M2/M3)"
    pip install torch torchvision torchaudio
else
    echo "  → Détection Mac Intel"
    pip install torch torchvision torchaudio
fi

# Installer LeRobot
echo ""
echo "🤖 Installation de LeRobot..."
pip install lerobot

# Installer les dépendances pour SmolVLA
echo ""
echo "🧠 Installation des dépendances SmolVLA..."
pip install transformers>=4.45.0
pip install datasets
pip install pillow
pip install opencv-python
pip install numpy
pip install scipy
pip install matplotlib
pip install tqdm
pip install pyyaml
pip install hydra-core
pip install omegaconf

# Installer wandb (optionnel, pour le tracking)
echo ""
echo "📊 Installation de Weights & Biases (optionnel)..."
pip install wandb

# Installer les outils de communication série pour SO100
echo ""
echo "🔌 Installation des outils pour SO100..."
pip install pyserial
pip install dynamixel-sdk

# Créer le fichier requirements.txt pour référence
echo ""
echo "📝 Création du fichier requirements.txt..."
pip freeze > requirements.txt

echo ""
echo "✅ Installation terminée avec succès!"
echo ""
echo "Pour activer l'environnement, utilise:"
echo "  source venv_smolvla/bin/activate"
echo ""
echo "Pour désactiver l'environnement:"
echo "  deactivate"
