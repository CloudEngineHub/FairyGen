import torch
from PIL import Image
from pathlib import Path
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth"),
    ],
)
pipe.load_lora(pipe.dit, "./lora-libraries/pig_walk/merged/200_400.safetensors", alpha=1)
input_image = Image.open("./data/pig_walk/shot/1.png").convert("RGB").resize((832, 480))
save_path = "./outputs/1.mp4"
Path(save_path).parent.mkdir(parents=True, exist_ok=True)

video = pipe(
    prompt="[p]_character_[w]_motion [p] walks towards the camera, filling the frame with a sense of movement. As [p] walks forward, the desks and classroom background shift backward, emphasizing the [p] progress through the space.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    input_image=input_image,
    num_frames=81,
    seed=1, tiled=True,
)
save_video(video, save_path, fps=15, quality=5)

