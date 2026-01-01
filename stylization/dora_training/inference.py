from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", dtype=torch.float16)
pipe.to("cuda")

# Load Lora weights
lora_path = "./lora-libraries/pig"
pipe.load_lora_weights(lora_path)

width = 720
height = 480

prompt ="A bustling city street with tall buildings, traffic, and street lights glowing at dusk in a childlike whimsical and illustrative style."
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5, height=height, width=width).images[0]
image.save("./outputs/city_street.png")