"""
Finetuning d'Octo sur le dataset SO-100 (so100_dataset.npz) généré par dataset_building.py.

Le robot SO-100 a 6 DOF. Le goal est une image de la pose finale (pince sur le cube).
L'action est le delta de qpos entre la pose initiale et la pose goal.

Usage :
    python finetune_octo.py \
        --pretrained_path=hf://rail-berkeley/octo-small-1.5 \
        --data_path=so100_dataset.npz \
        --save_dir=./octo_so100_finetuned
"""
from absl import app, flags, logging
import flax
import jax
import numpy as np
import optax
import tqdm
import wandb

from octo.model.components.action_heads import L1ActionHead
from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.spec import ModuleSpec
from octo.utils.train_utils import freeze_weights, merge_params, TrainState

FLAGS = flags.FLAGS

flags.DEFINE_string("pretrained_path", None, "Chemin vers le checkpoint Octo pré-entraîné.")
flags.DEFINE_string("data_path", "so100_dataset.npz", "Chemin vers so100_dataset.npz.")
flags.DEFINE_string("save_dir", None, "Répertoire de sauvegarde des checkpoints.")
flags.DEFINE_integer("batch_size", 32, "Taille du batch.")
flags.DEFINE_bool("freeze_transformer", False, "Geler les poids du transformer pré-entraîné.")

ACTION_DIM     = 6   # joints SO-100 : Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw
ACTION_HORIZON = 1   # on prédit un seul delta d'action
WINDOW_SIZE    = 2   # window_size du modèle octo-small-1.5
IMAGE_SIZE     = 256

class SO100Dataset:
    """Dataset numpy chargé depuis so100_dataset.npz."""

    def __init__(self, path: str):
        d = np.load(path)
        # Images en float32 [0, 1] pour Octo
        self.obs_images  = d["obs_images"].astype(np.float32)  / 255.0  # (N, 256, 256, 3)
        self.goal_images = d["goal_images"].astype(np.float32) / 255.0  # (N, 256, 256, 3)
        self.actions     = d["actions"].astype(np.float32)               # (N, 6)
        self.n = len(self.obs_images)
        logging.info(f"Dataset chargé : {self.n} épisodes depuis {path}")

    def get_batch(self, batch_size: int):
        """Retourne un batch aléatoire au format attendu par Octo.

        Octo-small a window_size=2 : on duplique l'obs unique sur les 2 timesteps.
        Shapes attendues :
          observation["image_primary"]  : (B, window_size, H, W, C)
          timestep_pad_mask             : (B, window_size)
          action                        : (B, window_size, action_horizon, action_dim)
          action_pad_mask               : même shape que action (4D)
        """
        idx = np.random.randint(0, self.n, size=batch_size)
        obs  = self.obs_images[idx]   # (B, H, W, C)
        goal = self.goal_images[idx]  # (B, H, W, C)
        act  = self.actions[idx]      # (B, 6)

        # Dupliquer sur window_size=2 (on n'a qu'un timestep par épisode)
        obs_win = np.stack([obs, obs], axis=1)            # (B, 2, H, W, C)
        act_win = np.stack([act, act], axis=1)            # (B, 2, 6)
        act_win = act_win[:, :, None, :]                  # (B, 2, 1, 6)

        return {
            "observation": {
                "image_primary":     obs_win,                                    # (B, 2, H, W, C)
                "timestep_pad_mask": np.ones((batch_size, WINDOW_SIZE), dtype=bool),  # (B, 2)
            },
            "task": {
                "image_primary": goal,                                           # (B, H, W, C)
            },
            "action":          act_win,                                          # (B, 2, 1, 6)
            "action_pad_mask": np.ones((batch_size, WINDOW_SIZE, ACTION_HORIZON, ACTION_DIM), dtype=bool),  # (B, 2, 1, 6)
        }


def main(_):
    assert FLAGS.batch_size % jax.device_count() == 0, \
        "Batch size must be divisible by device count."

    initialize_compilation_cache()

    wandb.init(name="finetune_so100", project="octo")

    # Chargement du modèle pré-entraîné
    logging.info("Chargement du modèle pré-entraîné...")
    pretrained_model = OctoModel.load_pretrained(FLAGS.pretrained_path)

    # Chargement du dataset SO-100
    dataset = SO100Dataset(FLAGS.data_path)

    # Modification de la config pour SO-100
    config = pretrained_model.config

    # Supprimer la caméra de poignet (absente sur SO-100)
    if "wrist" in config["model"]["observation_tokenizers"]:
        del config["model"]["observation_tokenizers"]["wrist"]

    # Supprimer le tokenizer proprio s'il existe (on n'utilise que l'image)
    if "proprio" in config["model"]["observation_tokenizers"]:
        del config["model"]["observation_tokenizers"]["proprio"]

    # Tête d'action L1 pour SO-100 : 6 DOF, horizon 1
    config["model"]["heads"]["action"] = ModuleSpec.create(
        L1ActionHead,
        action_horizon=ACTION_HORIZON,
        action_dim=ACTION_DIM,
        readout_key="readout_action",
    )

    # Création d'un batch exemple pour initialiser le modèle
    example_batch = dataset.get_batch(FLAGS.batch_size)

    logging.info("Initialisation du modèle adapté SO-100...")
    model = OctoModel.from_config(
        config,
        example_batch,
        text_processor=None,   # pas de langage
        verbose=True,
    )
    merged_params = merge_params(model.params, pretrained_model.params)
    model = model.replace(params=merged_params)
    del pretrained_model

    # Optimiseur
    learning_rate = optax.join_schedules(
        [optax.linear_schedule(0, 3e-5, 100), optax.constant_schedule(3e-5)], [100]
    )
    tx = optax.adamw(learning_rate)
    frozen_keys = model.config["optimizer"]["frozen_keys"]
    if FLAGS.freeze_transformer:
        frozen_keys.append("BlockTransformer_0")
    tx = freeze_weights(tx, model.params, frozen_keys)
    train_state = TrainState.create(
        rng=jax.random.PRNGKey(1234),
        model=model,
        tx=tx,
    )

    # Fonction de perte et étape d'entraînement
    def loss_fn(params, batch, rng, train=True):
        bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
        transformer_embeddings = bound_module.octo_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["timestep_pad_mask"],
            train=train,
        )
        action_loss, action_metrics = bound_module.heads["action"].loss(
            transformer_embeddings,
            batch["action"],
            batch["observation"]["timestep_pad_mask"],
            batch["action_pad_mask"],
            train=train,
        )
        return action_loss, action_metrics

    @jax.jit
    def train_step(state, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params, batch, dropout_rng, train=True
        )
        new_state = state.apply_gradients(grads=grads, rng=rng)
        return new_state, info

    # Boucle de finetuning
    logging.info("Démarrage du finetuning...")
    for i in tqdm.tqdm(range(5000), total=5000, dynamic_ncols=True):
        batch = dataset.get_batch(FLAGS.batch_size)
        train_state, update_info = train_step(train_state, batch)
        if (i + 1) % 100 == 0:
            update_info = jax.device_get(update_info)
            wandb.log(
                flax.traverse_util.flatten_dict({"training": update_info}, sep="/"),
                step=i,
            )
        if (i + 1) % 1000 == 0 and FLAGS.save_dir:
            train_state.model.save_pretrained(step=i, checkpoint_path=FLAGS.save_dir)


if __name__ == "__main__":
    app.run(main)