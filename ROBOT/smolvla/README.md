# SmolVLA pour SO100 - Guide Complet

Ce projet te permet d'entraîner et d'utiliser SmolVLA (Vision-Language-Action) sur ton robot SO100 de LeRobot.

## 📋 Prérequis

- **macOS** (testé sur macOS 12+)
- **Python 3.10+**
- **Robot SO100** de HuggingFace LeRobot
- **Webcam** ou caméra USB
- **8GB+ RAM** recommandé
- **GPU optionnel** (Apple Silicon MPS ou CUDA)

## 🚀 Installation

### 1. Cloner ou télécharger ce projet

```bash
cd ~/Documents  # ou le répertoire de ton choix
# Si téléchargé, extraire l'archive
cd smolvla_so100_project
```

### 2. Rendre le script d'installation exécutable

```bash
chmod +x setup_env.sh
```

### 3. Lancer l'installation

```bash
./setup_env.sh
```

Cette commande va :
- Créer un environnement virtuel `venv_smolvla`
- Installer PyTorch (optimisé pour ton Mac)
- Installer LeRobot et toutes les dépendances
- Installer SmolVLA et les outils nécessaires

⏱️ **Temps estimé** : 10-15 minutes

### 4. Activer l'environnement

```bash
source venv_smolvla/bin/activate
```

Tu verras `(venv_smolvla)` apparaître dans ton terminal.

## ⚙️ Configuration

### 1. Trouver le port de ton robot SO100

```bash
ls /dev/tty.*
```

Tu verras quelque chose comme : `/dev/tty.usbmodem14201`

### 2. Modifier la configuration

Ouvre `config.py` et modifie la ligne :

```python
SO100_CONFIG = {
    "port": "/dev/tty.usbmodem14201",  # ← REMPLACE PAR TON PORT
    ...
}
```

### 3. Configurer ta caméra

Si ta webcam n'est pas l'index 0, modifie dans `config.py` :

```python
CAMERA_CONFIG = {
    "top": {
        "index": 0,  # ← Change si nécessaire (essaie 0, 1, 2...)
        ...
    }
}
```

## 📖 Utilisation

### Workflow complet

```
1. Collecter des données → 2. Entraîner SmolVLA → 3. Tester
```

---

### 📹 Étape 1 : Collecter des données

**Prépare ton environnement :**
- Place des objets sur une table
- Assure-toi que la caméra voit bien la scène
- Connecte et allume ton SO100

**Lance la collecte :**

```bash
python 1_collect_data.py \
    --task "pick the red cube" \
    --num-episodes 50
```

**Options disponibles :**
- `--task` : Description de la tâche (obligatoire)
- `--num-episodes` : Nombre de démonstrations (défaut: 50)
- `--repo-id` : Pour sauvegarder sur HuggingFace (optionnel)

**Exemple avec HuggingFace :**

```bash
python 1_collect_data.py \
    --task "place object in box" \
    --num-episodes 30 \
    --repo-id ton-username/place-object-so100
```

**Conseils pour collecter :**
- ✅ Varie les positions de départ des objets
- ✅ Montre différentes approches
- ✅ Sois fluide dans tes mouvements
- ❌ Évite les mouvements trop rapides ou brusques

**Temps estimé :** 30-60 minutes pour 50 épisodes

---

### 🧠 Étape 2 : Entraîner SmolVLA

**Lance l'entraînement :**

```bash
python 2_train_smolvla.py \
    --dataset-path ./data/local/pick_the_red_cube_20240227_143022
```

**Avec Weights & Biases (pour suivre l'entraînement) :**

```bash
# D'abord se connecter à wandb
wandb login

# Puis lancer l'entraînement
python 2_train_smolvla.py \
    --dataset-path ./data/local/pick_the_red_cube_20240227_143022 \
    --use-wandb
```

**Surveiller l'entraînement :**
- Les logs s'affichent dans le terminal
- Si wandb est activé : ouvre https://wandb.ai dans ton navigateur
- Les checkpoints sont sauvegardés dans `./checkpoints/`

**Temps estimé :**
- Sans GPU : plusieurs heures
- Avec GPU (Apple Silicon) : 1-2 heures
- Avec GPU (CUDA) : 30min - 1h

**Interruption :** Tu peux arrêter avec `CTRL+C`, les checkpoints sont sauvegardés.

---

### 🧪 Étape 3 : Tester le modèle

#### Test simple (sans robot)

```bash
python 3_test_inference.py \
    --model-path ./checkpoints/smolvla_so100_20240227/final_model \
    --task "pick the red cube"
```

#### Visualiser les prédictions

```bash
python 3_test_inference.py \
    --model-path ./checkpoints/smolvla_so100_20240227/final_model \
    --task "pick the red cube" \
    --visualize
```

Cela génère `predictions_visualization.png`.

#### Test sur le robot réel

```bash
python 3_test_inference.py \
    --model-path ./checkpoints/smolvla_so100_20240227/final_model \
    --task "pick the red cube" \
    --robot \
    --duration 30
```

**Options :**
- `--duration` : Durée en secondes (défaut: 30)
- Appuie sur `CTRL+C` pour arrêter à tout moment

⚠️ **Sécurité** : Surveille toujours le robot pendant l'exécution !

---

## 📂 Structure du projet

```
smolvla_so100_project/
├── setup_env.sh           # Script d'installation
├── config.py              # Configuration (PORT, caméras, etc.)
├── 1_collect_data.py      # Collecter des démonstrations
├── 2_train_smolvla.py     # Entraîner SmolVLA
├── 3_test_inference.py    # Tester le modèle
├── README.md              # Ce fichier
├── venv_smolvla/          # Environnement virtuel (créé par setup)
├── data/                  # Datasets collectés
├── checkpoints/           # Modèles entraînés
└── logs/                  # Logs d'entraînement
```

## 🔧 Dépannage

### Le port du robot n'est pas trouvé

```bash
# Lister tous les ports
ls /dev/tty.*

# Si rien n'apparaît, vérifier que le robot est branché
# Essayer de le débrancher/rebrancher
```

### Erreur "permission denied"

```bash
# Sur Mac, tu peux avoir besoin des permissions
sudo chmod 666 /dev/tty.usbmodem*
```

### La caméra ne fonctionne pas

```bash
# Tester quelle caméra fonctionne
python -c "import cv2; print([cv2.VideoCapture(i).isOpened() for i in range(5)])"

# Modifier l'index dans config.py
```

### L'entraînement est trop lent

- Réduis `batch_size` dans `config.py` (essaie 4 ou 2)
- Réduis `num_epochs` pour un premier test (essaie 10-20)
- Si tu as un Mac M1/M2/M3, vérifie que MPS est activé

### Erreur "out of memory"

```python
# Dans config.py, réduire :
TRAINING_CONFIG = {
    "batch_size": 2,  # Au lieu de 8
    "gradient_accumulation_steps": 8,  # Au lieu de 4
}
```

### Le modèle ne performe pas bien

- Collecte plus de données (100+ épisodes recommandé)
- Varie davantage les conditions (positions, éclairage)
- Augmente la durée d'entraînement
- Vérifie que tes démonstrations sont de bonne qualité

## 📊 Métriques attendues

Avec 50 épisodes de bonne qualité :
- **Loss final** : < 0.1
- **Taux de succès** : 70-85%
- **Vitesse d'inférence** : 10-30 Hz

Avec 100+ épisodes :
- **Loss final** : < 0.05
- **Taux de succès** : 85-95%

## 🎯 Exemples de tâches

Tâches simples (50 épisodes suffisent) :
- ✅ "pick the red cube"
- ✅ "push the object forward"
- ✅ "grasp the bottle"

Tâches moyennes (100+ épisodes) :
- 🟡 "place the cube in the box"
- 🟡 "stack two blocks"
- 🟡 "open the drawer"

Tâches complexes (200+ épisodes) :
- 🔴 "sort objects by color"
- 🔴 "pour water into cup"
- 🔴 "fold the cloth"

## 💡 Conseils avancés

### Multi-tâches

Entraîne sur plusieurs tâches à la fois :

```bash
# Collecter plusieurs datasets
python 1_collect_data.py --task "pick cube" --num-episodes 50
python 1_collect_data.py --task "push object" --num-episodes 50

# Combiner et entraîner (nécessite modification du code)
```

### Fine-tuning incrémental

Tu peux continuer l'entraînement d'un modèle existant :

```bash
python 2_train_smolvla.py \
    --dataset-path ./data/new_task \
    --model-path ./checkpoints/previous_model/final_model
```

### Augmentation de données

Modifie `config.py` pour activer l'augmentation :

```python
TRAINING_CONFIG = {
    ...
    "augmentation": {
        "random_crop": True,
        "color_jitter": True,
        "random_rotation": 5,
    }
}
```

## 🔗 Ressources

- [LeRobot GitHub](https://github.com/huggingface/lerobot)
- [SmolVLM Documentation](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)
- [SO100 sur HuggingFace](https://huggingface.co/lerobot/so100)
- [Discord LeRobot](https://discord.gg/s3KuuzsPFb)

## 🐛 Signaler un bug

Si tu rencontres un problème :
1. Vérifie le dépannage ci-dessus
2. Ouvre une issue sur GitHub avec :
   - Logs d'erreur complets
   - Ton système (macOS version, Python version)
   - Étapes pour reproduire

## 📝 License

MIT License - Utilise et modifie librement !

---

Bon coding avec ton SO100 ! 🤖✨
