# Third-Party Repositories

External repositories are cloned here and kept separate from first-party code.

## Included repos

- `mujoco_menagerie/`
  - Source: https://github.com/google-deepmind/mujoco_menagerie
  - Used for robot XML and mesh assets (SO-100 model under `trs_so_arm100/`).

- `lerobot/`
  - Source: https://github.com/huggingface/lerobot
  - Used by sim-to-real servo/control integration.

## Update commands

```bash
cd "third_party/mujoco_menagerie" && git pull
cd "../lerobot" && git pull
```
