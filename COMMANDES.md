# Calibration
```bash
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem59700734041 --robot.id=my_awesome_follower_arm2
    ```

lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58FA0928531 \
    --teleop.id=my_awesome_leader_arm2


# Teleopération
```bash
lerobot-teleoperate --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem59700734041 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras='{ camera1: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, camera2: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30} }' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58FA0928531 \
    --teleop.id=my_awesome_leader_arm
    --display_data=true
```

# Record un dataset
```bash
HF_USER=$(NO_COLOR=1 hf auth whoami | awk -F': *' 'NR==1 {print $2}') && \
[[ -n "$HF_USER" ]] || { echo "HF_USER is empty. Run: hf auth login"; exit 1; } && \
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem59700734041 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ camera_side: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, camera_wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58FA0928531 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/Dataset_v1 \
    --dataset.num_episodes=16 \
    --dataset.single_task="Pick the cube and put it in the bin" \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2
```

# Review a dataset
```bash
lerobot-replay \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem59700734041 \
    --robot.id=my_awesome_follower_arm \
    --dataset.repo_id=${HF_USER}/pick-cube-640 \
    --dataset.episode=3 # choose the episode you want to replay
```

# Train a policy
```bash
HF_USER=$(NO_COLOR=1 hf auth whoami | awk -F': *' 'NR==1 {print $2}') && \
lerobot-train \
  --dataset.repo_id=${HF_USER}/pick-cube-640 \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --output_dir=outputs/train/pick-cube-640 \
  --job_name=act_pick-cube-640 \
  --policy.device=mps \
  --wandb.enable=false \
  --policy.repo_id=${HF_USER}/my_policy
```
# Eval la policy
```bash
!lerobot-eval
--policy.path=Christelle04/my_policy
--robot.type=so101_follower
--robot.port=/dev/tty.usbmodem59700734041
--robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}"
```

Ou 

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem59700734041 \
  --robot.id my_awesome_follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30}}" \
  --dataset.single_task="Grab the red cube and put it in the box" \
  --dataset.repo_id=${HF_USER}/eval_my_pick_place_test \
  --dataset.episode_time_s=50 \
  --dataset.num_episodes=10 \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --policy.path=${HF_USER}/my_pick_place
```


```bash
lerobot-record \
  --robot.type=so100_follower \
  --robot.port=/dev/tty.usbmodem59700734041 \
  --robot.cameras="{ camera_side: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, camera2_wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
  --robot.id=my_awesome_follower_arm \
  --display_data=true \
  --dataset.repo_id=${HF_USER}/eval_so100 \
  --dataset.single_task="Put lego brick into the transparent box" \
  --rename_map='{"observation.images.left": "observation.images.front", "observation.images.top": "observation.images.front"}' \
  --policy.path=${HF_USER}/my_pick_place
```


## Hugging face Tokens :
```bash
hf auth login --token "$HF_TOKEN" --add-to-git-credential
```

# Pour Hugging face : 
```bash
HF_USER=$(NO_COLOR=1 hf auth whoami | awk -F': *' 'NR==1 {print $2}')
echo $HF_USER
```

# Infos importantes
Limites des joints :
  Moteur 0: min=-1.92, max=1.92
  Moteur 1: min=-3.32, max=0.174
  Moteur 2: min=-0.174, max=3.14
  Moteur 3: min=-1.66, max=1.66
  Moteur 4: min=-2.79, max=2.79
  Moteur 5: min=-0.174, max=1.75



## Dans so100-mujoco-rl
```bash
cd so100-mujoco-rl
pixi run main -a PPO train -e Env01
pixi run main -a PPO -m models/Env01_PPO/best_model.zip test -e Env01
pixi run main -a PPO -m models/Env06_PPO/best_model.zip test -e Env06
pixi run main -a PPO -m models/Env06_PPO/Env06_PPO_cp__1400000_steps.zip test -e Env06
pixi run main -a PPO -m models/Env06_PPO/Env06_PPO_cp__680000_steps.zip test -e Env06
pixi run main -a SAC -m models/Env06_SAC/best_model.zip test -e Env06
```
## Dans jsp quoi
Pour lancer éval d'un modèle SAC trained to reach le cube
mjpython eval_reach_sac.py --model outputs/reach_sac/best_model.zip \
                            --vecnorm outputs/reach_sac/vecnorm.pkl \
                            --close_only


## Dans SEPARATE MODELS : 
Train : 
```bash
python Train.py --task PICK --resume models/so100_pick_v1.zip
````
 OU 
````bash
python Train.py --task PICK
````

Voir résultat : 
```bash
mjpython Task_manager.py --pick models/so100_pick_v1.zip \
                       --place models/so100_place_v1.zip \
                       --episodes 5 --speed 1 --debug-reward
```

Ou

```bash
mjpython Task_manager.py --pick models/best_pick/best_model.zip --place models/best_place/best_model.zip --episodes 5 --speed 1 --debug-reward
```

#### Avec V3 rewards : 
```bash
# Entraîner les 3 modèles séquentiellement
python Train.py --task ALL_V3

# Ou individuellement
python Train.py --task REACH
python Train.py --task GRASP
python Train.py --task CARRY

# Visualiser la chaîne complète
mjpython Task_manager_v3.py \
    --reach models/best_reach/best_model.zip \
    --grasp models/best_grasp/best_model.zip \
    --carry models/best_carry/best_model.zip \
    --episodes 5 --speed 1 --debug-reward
```




# Entrainer une policy sur un dataset huggingface
```bash
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=LucaFrat/so100_mujoco_pick_red_cube \
  --policy.push_to_hub=false
```


## Dans SAC-THREE-AGENTS

#### Avec scène au sol
```bash
python train.py --task all       # train Reach → Grasp → Place with curriculum
mjpython task_manager.py --episodes 5 --render   # visualize full pipeline
python evaluate.py --mode full --episodes 100
```

#### Avec scène avec boîte
```bash
python train.py --task place --scene box   # Entraîner Place pour la scène boîte
mjpython task_manager.py --render --scene box --episodes 3 --debug # Visualiser avec rendu
```

### Possibilité d'ajouter --debug pour vérifier le comportement
-> Bons modèles sauvegardés dans saves_models


## Fine tuner SmolVLA
```bash
python lerobot/scripts/train.py \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=REPO_ID \
  --batch_size=64 \
  --steps=20000  # 10% of training budget
```
```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/<dataset> \
  --output_dir=./outputs/[RUN_NAME] \
  --job_name=[RUN_NAME] \
  --policy.repo_id=${HF_USER}/<desired_policy_repo_id> \
  --policy.path=lerobot/[BASE_CHECKPOINT] \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --steps=100000 \
  --batch_size=4
```

```bash
mjpython Task_manager.py --pick models/so100_pick_v9_r-v2.zip \
                       --place models/so100_place_v1.zip \
                       --episodes 5 --speed 1 --debug-reward
```





```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem59700734041 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ camera1: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, camera2: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58FA0928531 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/pick_place \
    --dataset.num_episodes=50 \
    --dataset.single_task="Grab the red cube and put it in the box" \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2 \
    --resume=true
```

```bash
HF_USER=$(NO_COLOR=1 hf auth whoami | awk -F': *' 'NR==1 {print $2}') && \
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem59700734041 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ camera1: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, camera2: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58FA0928531 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/Dataset_v3 \
    --dataset.num_episodes=20 \
    --dataset.single_task="Grab the red cube and put it in the box" \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2 \
    --resume=false
```



Given by Copilot after training on the mac studio : 
```bash
lerobot-record \
  --robot.type=so100_follower \
  --robot.port=/dev/tty.usbmodem59700734041 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras='{ camera1: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, camera2: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30} }' \
  --dataset.repo_id=Christelle04/eval_ACT2_v3 \
  --dataset.single_task="Pick and place evaluation" \
  --dataset.num_episodes=5 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=10 \
  --dataset.fps=30 \
  --dataset.push_to_hub=false \
  --policy.path=Christelle04/ACT2
  ```


To delete the old file (because of problems and errors) and start fresh : 
```bash
set -e
TARGET="$HOME/.cache/huggingface/lerobot/Christelle04/eval_ACT2_v3"
if [[ -d "$TARGET" ]]; then
  rm -rf "$TARGET"
fi
if [[ -e "$TARGET" ]]; then
  echo "STILL_EXISTS"
else
  echo "DELETED_OK"
fi
```

To upload dataset: 
```bash
hf upload-large-folder Christelle04/eval_ACT2_v3 "$HOME/.cache/huggingface/lerobot/Christelle04/eval_ACT2_v3" --repo-type dataset
```
To upload policy : 
```bash
hf upload Christelle04/ACT2 \
~/outputs/train/ACT2/checkpoints/last/pretrained_model \    
. \
--repo-type=model
  ```


Reprendre le training jusqu'à Xk steps : 
```bash
lerobot-train \                                              
  --config_path=outputs/train/ACT2/checkpoints/last/pretrained_model/train_config.json \
  --resume=true \
  --steps=X --save_freq=5000
  ```

To run the training : 
```bash
lerobot-train \                                              
  --dataset.repo_id=Christelle04/pick_place \                                           
  --policy.type=act \
  --policy.device=mps \         
  --policy.push_to_hub=false \
  --output_dir=outputs/train/ACT2 \
  --job_name=ACT2 \
  --wandb.enable=false \
  --steps=20000 \
  --save_freq=5000
  ```





Run the eval with a teleop : 
```bash
lerobot-record \
  --robot.type=so100_follower \
  --robot.port=/dev/tty.usbmodem59700734041 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras='{ camera1: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, camera2: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30} }' \
  --dataset.repo_id=Christelle04/eval_ACT2_v3 \
  --teleop.type=so100_leader \
  --teleop.port=/dev/tty.usbmodem58FA0928531 \
  --teleop.id=my_awesome_leader_arm \
  --dataset.single_task="Pick and place evaluation" \
  --dataset.num_episodes=5 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=10 \
  --dataset.fps=30 \
  --dataset.push_to_hub=false \
  --policy.path=Christelle04/ACT2
```



------------------------------------------CLEAN COMMANDS TO KEEP---------------------------------

## Calibration
```bash
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem59700734041 \
    --robot.id=my_awesome_follower_arm
````

```bash
lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58FA0928531 \
    --teleop.id=my_awesome_leader_arm
```

## Téléopération avec les deux cameras :
```bash
lerobot-teleoperate --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem59700734041 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ camera_side: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, camera_wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58FA0928531 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true
```

## Record un dataset avec les bonnes infos :
```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem59700734041 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ camera_side: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, camera_wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58FA0928531 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.repo_id=Christelle04/Dataset_v3 \
    --dataset.num_episodes=20 \
    --dataset.single_task="Pick the cube and put it in the bin" \
    --resume=true --dataset.push_to_hub=false
```

## Uploader le dataset sur Hugging face : 
```bash
hf upload Christelle04/DATASET_NAME "$HOME/.cache/huggingface/lerobot/Christelle04/DATASET_NAME" --repo-type dataset
```


## Sur une nouvelle instance, commandes à faire pour tt installer et lancer le training : 

```bash
# Paquets système
sudo apt-get update && sudo apt-get install -y git git-lfs ffmpeg python3-pip tmux
git lfs install

# Drivers NVIDIA + reboot
sudo apt install -y nvidia-driver-580
sudo reboot
# (reconnexion après ~1 min, puis vérifier)
nvidia-smi

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
git clone https://github.com/huggingface/lerobot.git
cd lerobot
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[feetech,smolvla,training,dataset]"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

hf auth login            # token WRITE
uv pip install wandb && wandb login``

python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Christelle04/Dataset_v3', repo_type='dataset', local_dir='/home/ubuntu/dataset_v3')
"
ls /home/ubuntu/dataset_v3/meta/
df -h /

tmux new -s act
cd ~/lerobot && source .venv/bin/activate
```

## For ACT with 1 GPU : 
```bash
lerobot-train --dataset.repo_id=Christelle04/Dataset_v3 --dataset.root=/home/ubuntu/dataset_v3 --policy.type=act --policy.device=cuda --policy.push_to_hub=true --policy.repo_id=Christelle04/act_dataset_v3_clean --output_dir=outputs/train/act_clean --job_name=act_clean --wandb.enable=true --batch_size=8 --steps=100000 --save_freq=10000 && sudo shutdown -h now
```


## Training de ACT sur ce dataset (sur instance AWS Ubuntu avec 4 GPU) : 
```bash
accelerate launch --multi_gpu --num_processes=4 --mixed_precision=fp16 $(which lerobot-train) --dataset.repo_id=Christelle04/Dataset_v3 --dataset.root= /home/ubuntu/dataset_v3 --policy.type=act --policy.push_to_hub=true --policy.repo_id=Christelle04/act_dataset_v3 --output_dir=outputs/train/ACT_dataset_v3 --job_name=ACT_dataset_v3 --wandb.enable=true --batch_size=32 --steps=100000 --save_freq=5000 --optimizer.lr=4e-5 && sudo shutdown -h now
```

## Train de SmolVLA sur ce dataset (sur instance AWS Ubuntu avec 1 GPU) : 
```bash
lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id=Christelle04/Dataset_v4 --dataset.root=/home/ubuntu/dataset_v4 --rename_map='{"observation.images.camera_side": "observation.images.camera1", "observation.images.camera_wrist": "observation.images.camera2"}' --policy.device=cuda --policy.push_to_hub=true --policy.repo_id=Christelle04/smolvla_dataset_v4 --output_dir=outputs/train/smolvla_v4 --job_name=smolvla_v4 --wandb.enable=true --wandb.disable_artifact=true --batch_size=16 --steps=20000 --save_freq=20000 && sudo shutdown -h now
```

## Train de SmolVLA sur ce dataset (sur instance AWS Ubuntu avec 4 GPU) : 
```bash
accelerate launch --multi_gpu --num_processes=4 --mixed_precision=bf16 $(which lerobot-train) --policy.path=lerobot/smolvla_base --dataset.repo_id=Christelle04/Dataset_v4 --dataset.root=/home/ubuntu/dataset_v4 --rename_map='{"observation.images.camera_side": "observation.images.camera1", "observation.images.camera_wrist": "observation.images.camera2"}' --policy.push_to_hub=true --policy.repo_id=Christelle04/smolvla_dataset_v4 --output_dir=outputs/train/smolvla_v4 --job_name=smolvla_v4 --wandb.enable=true --wandb.disable_artifact=true --batch_size=16 --steps=20000 --save_freq=20000 && sudo shutdown -h now
```



## Évaluation de la policy :
```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem59700734041 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ camera_side: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, camera_wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58FA0928531 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.repo_id=Christelle04/eval_POLICY_DATASETVERSION_NUMSTEPS \
    --dataset.single_task="Evaluation of POLICY_TO_EVAL on DATASETVERSION_NUMSTEPS" \
    --dataset.num_episodes=5 \
    --dataset.episode_time_s=60 \
    --dataset.reset_time_s=15 \
    --dataset.fps=30 \
    --dataset.push_to_hub=false \
    --policy.path=Christelle04/POLICY_TO_EVAL
```



RECAP install sur nouvelle instance EC2 : 
Étape 2 — Paquets système
sudo apt-get update && sudo apt-get install -y git git-lfs ffmpeg python3-pip tmux
git lfs install
Étape 3 — Drivers NVIDIA (puis reboot)
sudo apt install -y nvidia-driver-580
sudo reboot
# attendre ~1 min, se reconnecter, puis vérifier :
nvidia-smi
Étape 4 — Installer uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
Étape 5 — Cloner + installer lerobot
git clone https://github.com/huggingface/lerobot.git
cd lerobot
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[feetech,smolvla,training]"   # feetech=SO-100, smolvla, training=multi-GPU
Étape 6 — Vérifier GPU + PyTorch
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
Étape 7 — Authentification HuggingFace + wandb
hf auth login          # token avec accès WRITE
pip install wandb && wandb login   # clé sur https://wandb.ai/authorize
Étape 8 — Pré-télécharger le dataset (évite le bug de cache)
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Christelle04/Dataset_v3', repo_type='dataset', local_dir='/home/ubuntu/dataset_v3')
"
ls /home/ubuntu/dataset_v3/meta/   # doit contenir info.json, tasks.parquet, stats.json
df -h                               # vérifier l'espace libre
Étape 9 — Entraînement dans tmux
tmux new -s train
cd ~/lerobot && source .venv/bin/activate
ACT — multi-GPU (4× A10G) :

accelerate launch --multi_gpu --num_processes=4 --mixed_precision=fp16 $(which lerobot-train) --dataset.repo_id=Christelle04/Dataset_v3 --dataset.root=/home/ubuntu/dataset_v3 --policy.type=act --policy.device=cuda --policy.push_to_hub=true --policy.repo_id=Christelle04/act_dataset_v3 --output_dir=outputs/train/ACT_dataset_v3 --job_name=ACT_dataset_v3 --wandb.enable=true --batch_size=32 --steps=100000 --save_freq=5000 --optimizer.lr=4e-5 && sudo shutdown -h now
ACT — mono-GPU (g5.2xlarge) :

lerobot-train --dataset.repo_id=Christelle04/Dataset_v3 --dataset.root=/home/ubuntu/dataset_v3 --policy.type=act --policy.device=cuda --policy.use_amp=true --policy.push_to_hub=true --policy.repo_id=Christelle04/act_dataset_v3 --output_dir=outputs/train/ACT_dataset_v3 --job_name=ACT_dataset_v3 --wandb.enable=true --batch_size=32 --steps=100000 --save_freq=5000 && sudo shutdown -h now
SmolVLA — multi-GPU (⚠️ bf16, pas fp16, + rename_map des caméras) :

accelerate launch --multi_gpu --num_processes=4 --mixed_precision=bf16 $(which lerobot-train) --policy.path=lerobot/smolvla_base --dataset.repo_id=Christelle04/Dataset_v3 --dataset.root=/home/ubuntu/dataset_v3 --rename_map='{"observation.images.camera_side": "observation.images.camera1", "observation.images.camera_wrist": "observation.images.camera2"}' --policy.push_to_hub=true --policy.repo_id=Christelle04/smolvla_dataset_v3 --output_dir=outputs/train/smolvla_v3 --job_name=smolvla_v3 --wandb.enable=true --wandb.disable_artifact=true --batch_size=16 --steps=20000 --save_freq=5000 && sudo shutdown -h now
Détacher de tmux : Ctrl+B puis D — rattacher : tmux attach -t train

Étape 10 — Suivi
Loss en direct : wandb.ai → projet lerobot
GPU : watch -n 2 nvidia-smi (autre terminal)
Étape 11 — Si l'upload auto échoue (disque plein)
hf upload Christelle04/act_dataset_v3 ~/lerobot/outputs/train/ACT_dataset_v3/checkpoints/last/pretrained_model --repo-type=model
Étape 12 — Fin
L'instance s'arrête seule (&& sudo shutdown). Pense à Terminate dans la console EC2 pour stopper tous les frais.

Mémo des pièges rencontrés
Problème	Solution
apt-get not found	Amazon Linux → utiliser Ubuntu AMI
Timeout sur téléchargements	Ajouter règle 443 sortant au security group
externally-managed-environment	Installer uv via script, pas pip
requires Python>=3.12	uv venv --python 3.12
No NVIDIA driver	sudo apt install nvidia-driver-580 + reboot
info.json/tasks.parquet not found	Pré-télécharger avec snapshot_download + --dataset.root
Killed (OOM RAM)	Baisser --batch_size ou instance avec + de RAM
No space left	Volume ≥ 50 GB + --wandb.disable_artifact=true
SmolVLA BFloat16 not implemented	--mixed_precision=bf16 (pas fp16)
SmolVLA Feature mismatch caméras	--rename_map vers camera1/camera2
Output directory exists	rm -rf le dossier ou changer --output_dir