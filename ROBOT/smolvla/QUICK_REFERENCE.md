# Commandes Rapides - SmolVLA SO100

## Installation (une seule fois)

```bash
# 1. Rendre le script exécutable
chmod +x setup_env.sh

# 2. Lancer l'installation
./setup_env.sh

# 3. Activer l'environnement
source venv_smolvla/bin/activate
```

## Configuration initiale

```bash
# Trouver ton port SO100
ls /dev/tty.*

# Éditer config.py avec ton port
nano config.py  # ou ton éditeur préféré
```

## Workflow complet

### 1. Collecter des données

```bash
# Activer l'environnement
source venv_smolvla/bin/activate

# Collecter 50 épisodes
python 1_collect_data.py --task "pick the red cube" --num-episodes 50
```

### 2. Entraîner

```bash
# Sans wandb
python 2_train_smolvla.py --dataset-path ./data/local/pick_the_red_cube_XXXXXX

# Avec wandb (tracking)
wandb login
python 2_train_smolvla.py --dataset-path ./data/local/pick_the_red_cube_XXXXXX --use-wandb
```

### 3. Tester

```bash
# Test simple
python 3_test_inference.py --model-path ./checkpoints/smolvla_so100_XXXXXX/final_model

# Test avec visualisation
python 3_test_inference.py --model-path ./checkpoints/smolvla_so100_XXXXXX/final_model --visualize

# Test sur robot réel
python 3_test_inference.py --model-path ./checkpoints/smolvla_so100_XXXXXX/final_model --robot --duration 30
```

## Commandes utiles

### Environnement virtuel

```bash
# Activer
source venv_smolvla/bin/activate

# Désactiver
deactivate

# Vérifier les packages installés
pip list
```

### Debugging

```bash
# Lister les ports série
ls /dev/tty.*

# Tester les caméras disponibles
python -c "import cv2; print([i for i in range(5) if cv2.VideoCapture(i).isOpened()])"

# Vérifier PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'MPS: {torch.backends.mps.is_available()}')"
```

### Gestion des données

```bash
# Lister les datasets
ls -la data/local/

# Voir la taille des datasets
du -sh data/local/*

# Nettoyer les anciens checkpoints
rm -rf checkpoints/smolvla_so100_ANCIEN/
```

## Tips & Tricks

### Réinstaller proprement

```bash
# Supprimer l'environnement
rm -rf venv_smolvla/

# Relancer l'installation
./setup_env.sh
```

### Collecter en continue

```bash
# Script pour collecter plusieurs tâches d'affilée
for task in "pick cube" "push object" "grasp bottle"; do
    python 1_collect_data.py --task "$task" --num-episodes 30
done
```

### Monitoring GPU pendant l'entraînement

```bash
# Ouvrir un nouveau terminal et lancer :
watch -n 1 nvidia-smi  # Pour NVIDIA GPU

# Ou pour Mac M1/M2/M3 :
sudo powermetrics --samplers gpu_power -i 1000
```

### Interrompre proprement

- **CTRL+C** : Arrêt propre (sauvegarde les checkpoints)
- **CTRL+Z** puis `kill %1` : Force l'arrêt

## Erreurs fréquentes et solutions

### "ModuleNotFoundError: No module named 'lerobot'"

```bash
source venv_smolvla/bin/activate
```

### "Permission denied" sur /dev/tty.xxx

```bash
sudo chmod 666 /dev/tty.usbmodem*
```

### "CUDA out of memory"

Édite `config.py` :
```python
TRAINING_CONFIG = {
    "batch_size": 2,  # Réduit de 8 à 2
}
```

### Caméra ne fonctionne pas

```bash
# Teste quel index fonctionne
python -c "import cv2; [print(f'Cam {i}: {cv2.VideoCapture(i).isOpened()}') for i in range(5)]"

# Modifie dans config.py
```

## Variables d'environnement utiles

```bash
# Désactiver le cache HuggingFace (si problème de téléchargement)
export HF_HUB_OFFLINE=1

# Forcer CPU (pour debug)
export CUDA_VISIBLE_DEVICES=""

# Logs plus verbeux
export TRANSFORMERS_VERBOSITY=debug
```

## Backup et versioning

```bash
# Sauvegarder un modèle entraîné
cp -r checkpoints/smolvla_so100_XXXXXX ~/Backups/

# Versionner avec git (optionnel)
git init
git add *.py config.py README.md
git commit -m "Initial commit"
```

## Raccourcis shell (optionnel)

Ajoute à `~/.zshrc` ou `~/.bashrc` :

```bash
# Alias pour activer l'environnement
alias smolvla="cd ~/path/to/smolvla_so100_project && source venv_smolvla/bin/activate"

# Alias pour les scripts
alias collect="python 1_collect_data.py"
alias train="python 2_train_smolvla.py"
alias test="python 3_test_inference.py"
```

Puis : `source ~/.zshrc`

Usage : `smolvla` puis `collect --task "pick cube" --num-episodes 50`
