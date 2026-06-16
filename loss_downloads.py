import wandb
import pandas as pd
from pathlib import Path

ENTITY = "christelle-nollet-octo-technology"
PROJECT = "lerobot"
OUT_DIR = Path("wandb_loss_exports")

api = wandb.Api()
OUT_DIR.mkdir(exist_ok=True)

runs = list(api.runs(f"{ENTITY}/{PROJECT}"))
print(f"Runs trouvés: {len(runs)}")

for run in runs:
    print(f"\nRun: {run.name} | id={run.id} | state={run.state}")

    # récupère un échantillon pour voir les noms exacts des métriques
    sample = run.history(samples=10, pandas=True)
    print("Colonnes trouvées:", list(sample.columns))

    loss_keys = [
        c for c in sample.columns
        if "loss" in c.lower()
    ]

    if not loss_keys:
        print("Aucune colonne contenant 'loss'")
        continue

    print("Loss keys:", loss_keys)

    rows = list(run.scan_history(keys=["_step", *loss_keys]))
    df = pd.DataFrame(rows)

    safe_name = (run.name or run.id).replace("/", "_").replace(" ", "_")
    out_file = OUT_DIR / f"{safe_name}_{run.id}_loss.csv"

    df.to_csv(out_file, index=False)
    print(f"Exporté: {out_file}")