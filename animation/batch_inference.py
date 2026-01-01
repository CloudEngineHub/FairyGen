import torch
import os
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

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

input_folder = "./data/pig_walk/shot/"
output_folder = "./outputs/pig_walk/"
os.makedirs(output_folder, exist_ok=True)

negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

image_files = [f for f in os.listdir(input_folder) if f.endswith(".png")]
print(f"find {len(image_files)} images, start processing...")

for img_name in sorted(image_files):
    base_name = os.path.splitext(img_name)[0]  # 获取不带后缀的文件名，如 "1"
    txt_name = f"{base_name}.txt"
    
    img_path = os.path.join(input_folder, img_name)
    txt_path = os.path.join(input_folder, txt_name)

    if not os.path.exists(txt_path):
        print(f"Skip: can not find corresponding txt file: {txt_name}")
        continue
    
    print(f"Processing: {img_name}...")

    input_image = Image.open(img_path).convert("RGB").resize((832, 480))

    with open(txt_path, "r", encoding="utf-8") as f:
        current_prompt = f.read().strip()

    video = pipe(
        prompt=current_prompt,
        negative_prompt=negative_prompt,
        input_image=input_image,
        num_frames=81,
        seed=1, 
        tiled=True,
    )

    output_path = os.path.join(output_folder, f"{base_name}.mp4")
    save_video(video, output_path, fps=15, quality=5)
    print(f"已保存: {output_path}")

print("所有批量任务处理完成！")
