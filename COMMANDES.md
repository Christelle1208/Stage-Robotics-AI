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
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58FA0928531 \
    --teleop.id=my_awesome_leader_arm
```

# Record un dataset
```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem59700734041 \
    --robot.id=my_awesome_follower_arm2 \
    --robot.cameras="{ camera1: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, camera2: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58FA0928531 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/pick_place \
    --dataset.num_episodes=50 \
    --dataset.single_task="Grab the red cube and put it in the box" \
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
  --robot.cameras="{ up: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30}, side: {type: intelrealsense, serial_number_or_name: 233522074606, width: 640, height: 480, fps: 30}}" \
  --robot.id=my_awesome_follower_arm \
  --display_data=false \
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

### API KEY WEIGHT AND BIASES :
 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


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
    --dataset.repo_id=${HF_USER}/pick_place \
    --dataset.num_episodes=10 \
    --dataset.single_task="Grab the red cube and put it in the box" \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2 \
    --resume=true
```