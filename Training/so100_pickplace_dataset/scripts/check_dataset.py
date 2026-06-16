from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("local/so100_mujoco_pickplace")
print(dataset)
print(dataset[0].keys())
print(dataset[0]["observation.state"].shape)
print(dataset[0]["action"].shape)