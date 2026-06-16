import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.act.modeling_act import ACTPolicy

POLICY_ID = "Christelle04/act_so100_test"
DATASET_ID = "Christelle04/pick_place"

device = torch.device("mps")  # ou "cpu"

policy = ACTPolicy.from_pretrained(POLICY_ID)
policy.to(device)
policy.eval()

metadata = LeRobotDatasetMetadata(DATASET_ID)
preprocess, postprocess = make_pre_post_processors(
    policy.config,
    dataset_stats=metadata.stats,
)

dataset = LeRobotDataset(DATASET_ID)

# Stats utilisées pour normaliser l'action
action_stats = metadata.stats["action"]
action_mean = torch.tensor(action_stats["mean"], dtype=torch.float32)
action_std = torch.tensor(action_stats["std"], dtype=torch.float32)

print("Action mean:", action_mean)
print("Action std:", action_std)

indices_to_test = [0, 50, 100, 200, 500, 1000]

for i in indices_to_test:
    if i >= len(dataset):
        continue

    sample = dataset[i]

    obs = {
        "observation.state": sample["observation.state"].unsqueeze(0).to(device),
        "observation.images.camera1": sample["observation.images.camera1"].unsqueeze(0).to(device),
        "observation.images.camera2": sample["observation.images.camera2"].unsqueeze(0).to(device),
    }

    gt_action = sample["action"].unsqueeze(0)

    policy.reset()

    with torch.inference_mode():
        obs_proc = preprocess(obs)

        # Action prédite dans l'espace normalisé
        pred_raw = policy.select_action(obs_proc)

        # Action prédite dénormalisée
        pred = postprocess(pred_raw)

    # Comparaison dans l'espace réel
    gt_real = gt_action.to(pred.device)
    mae_real = (pred - gt_real).abs().mean().item()

    # Comparaison dans l'espace normalisé
    gt_norm = (gt_action.cpu() - action_mean) / action_std
    mae_norm = (pred_raw.cpu() - gt_norm).abs().mean().item()

    print("\n" + "=" * 80)
    print(f"Index dataset: {i}")

    print("\nAction prédite postprocessée:")
    print(pred.cpu().numpy().round(3))

    print("\nAction vérité terrain:")
    print(gt_real.cpu().numpy().round(3))

    print("\nErreur absolue réelle:")
    print((pred - gt_real).abs().cpu().numpy().round(3))

    print("\nMAE réelle:")
    print(round(mae_real, 4))

    print("\nAction prédite RAW normalisée:")
    print(pred_raw.cpu().numpy().round(3))

    print("\nGT normalisée manuellement:")
    print(gt_norm.numpy().round(3))

    print("\nErreur absolue normalisée:")
    print((pred_raw.cpu() - gt_norm).abs().numpy().round(3))

    print("\nMAE normalisée:")
    print(round(mae_norm, 4))

    print("\nComparaison avec les 20 prochains timesteps:")
    max_offset = min(20, len(dataset) - i)

    for offset in range(max_offset):
        gt_future = dataset[i + offset]["action"].unsqueeze(0).to(pred.device)
        mae_future = (pred - gt_future).abs().mean().item()

        print(
            f"i+{offset:02d} | "
            f"MAE={mae_future:.3f} | "
            f"gt={gt_future.cpu().numpy().round(2)}"
        )
        
batch = {
    "observation.state": sample["observation.state"].unsqueeze(0).to(device),
    "observation.images.camera1": sample["observation.images.camera1"].unsqueeze(0).to(device),
    "observation.images.camera2": sample["observation.images.camera2"].unsqueeze(0).to(device),
    "action": sample["action"].unsqueeze(0).to(device),
    "action_is_pad": torch.zeros((1,), dtype=torch.bool, device=device),
}

with torch.inference_mode():
    batch_proc = preprocess(batch)

    # Forward bas niveau : récupère directement les actions prédites
    actions_hat, _, _ = policy.model(
        batch_proc["observation.images.camera1"],
        batch_proc["observation.images.camera2"],
        batch_proc["observation.state"],
        batch_proc["action"],
        batch_proc["action_is_pad"],
    )

    # Loss L1 comme dans ACT, sans la partie KL qui plante chez toi
    l1 = torch.nn.functional.l1_loss(
        batch_proc["action"],
        actions_hat,
        reduction="none",
    )

    mask = ~batch_proc["action_is_pad"].unsqueeze(-1)
    loss_l1 = (l1 * mask).mean()

print("Loss L1 normalisée:", loss_l1.item())
