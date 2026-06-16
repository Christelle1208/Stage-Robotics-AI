import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

repo_id = "allenai/MolmoAct2-SO100_101"

top_rgb = Image.open(
    hf_hub_download(repo_id, "assets/sample_realsense_top_rgb.png")
).convert("RGB")
side_rgb = Image.open(
    hf_hub_download(repo_id, "assets/sample_realsense_side_rgb.png")
).convert("RGB")
task = "Move the arm towards the lemon, grasp it, lift it up, and drop it into the red bowl."
robot_state = np.array(
    [
        -0.52734375,
        189.140625,
        181.40625,
        60.64453125,
        -3.603515625,
        1.0971786975860596,
    ],
    dtype=np.float32,
)

processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    repo_id,
    trust_remote_code=True,
    dtype=torch.bfloat16,
).to("cuda").eval()

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    out = model.predict_action(
        processor=processor,
        images=[top_rgb, side_rgb],
        task=task,
        state=robot_state,
        norm_tag="so100_so101_molmoact2",
        inference_action_mode="continuous",
        enable_depth_reasoning=False,
        num_steps=10,
        normalize_language=True,
        enable_cuda_graph=True,
    )

actions = out.actions
print("Predicted actions:", actions)
