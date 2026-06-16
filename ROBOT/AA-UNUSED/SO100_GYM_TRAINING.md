# SO100 Gymnasium + RL (Pick-and-Place)

Ce workspace contient maintenant une base prête pour entraîner une policy sur SO100 en simulation MuJoCo.

## Fichiers ajoutés

- `so100_rl/so100_pick_place_env.py`: environnement Gymnasium `SO100PickPlaceEnv`
- `mujoco_menagerie/trs_so_arm100/scene_pick_place.xml`: scène MuJoCo (robot + cube + cible)
- `train_so100_pick_place.py`: entraînement SB3 (PPO ou SAC)
- `eval_so100_pick_place.py`: évaluation d'un modèle entraîné
- `run_so100_random.py`: smoke test rapide avec policy aléatoire
- `requirements_so100_rl.txt`: dépendances minimales

## Tâches disponibles

- `reach`: atteindre le cube
- `pick`: saisir et lever le cube
- `pick_place`: saisir et poser sur la cible

## Installation

```bash
conda activate base
pip install -r requirements_so100_rl.txt
```

## Vérification rapide

```bash
python run_so100_random.py --task pick_place --episodes 3
```

Visualisation en direct (fenêtre MuJoCo):

```bash
python run_so100_random.py --task pick_place --episodes 3 --render_human
```

## Entraînement PPO

```bash
python train_so100_pick_place.py \
  --task pick_place \
  --algo ppo \
  --total_timesteps 300000 \
  --n_envs 4 \
  --assist_grasp
```

## Entraînement SAC

```bash
python train_so100_pick_place.py \
  --task pick_place \
  --algo sac \
  --total_timesteps 500000 \
  --n_envs 1 \
  --assist_grasp
```

## Évaluation

```bash
python eval_so100_pick_place.py \
  --model outputs/so100_rl/<run>/best_model/best_model.zip \
  --algo ppo \
  --task pick_place \
  --episodes 20 \
  --render_human
```

Pour enregistrer un GIF:

```bash
python eval_so100_pick_place.py \
  --model outputs/so100_rl/<run>/best_model/best_model.zip \
  --algo ppo \
  --task pick_place \
  --episodes 5 \
  --save_gif outputs/so100_rl/eval.gif
```

## Notes pratiques

- `--assist_grasp` active une aide de saisie (plus facile pour démarrer en RL).
- `--no_assist_grasp` permet de passer en mode plus réaliste ensuite.
- Les scripts supposent que tu les lances depuis la racine `ROBOT`.
