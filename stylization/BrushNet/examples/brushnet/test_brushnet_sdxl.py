import os
from diffusers import StableDiffusionXLBrushNetPipeline, BrushNetModel, DPMSolverMultistepScheduler, AutoencoderKL, \
    UNet2DConditionModel
import torch
import cv2
import numpy as np
from PIL import Image
from peft import LoraConfig, set_peft_model_state_dict
from safetensors.torch import load_file
from types import MethodType
from typing import Any

# Model paths
base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
brushnet_path = "/home/share/segmentation_mask_brushnet_ckpt_sdxl_v1"  # download from https://drive.google.com/drive/folders/1KBr71RlQEACJPcs2Uoanpi919nISpG1L
vae_path = "madebyollin/sdxl-vae-fp16-fix"

# Input paths
image_path = "./data/pig/first_frame/Walk.jpg"
mask_path  = "./data/pig/mask/mask.jpg"
prompt_dir = "./data/pig/prompt/Walk"

output_dir = "./outputs/pig"
lora_path  = "../dora_training/lora-libraries/pig/"
os.makedirs(output_dir, exist_ok=True)

# Conditioning scale
brushnet_conditioning_scale = 0.6  # adjust lora_scale to balance style strength and content alignment

def make_new_forward(name):
    def new_forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)
        mask_latents = kwargs.pop("mask_latents", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            lora_A_keys = self.lora_A.keys()
            for active_adapter in self.active_adapters:
                if active_adapter not in lora_A_keys:
                    continue

                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to("cuda")

                if not self.use_dora[active_adapter]:
                    update_input = dropout(x)
                    # apply mask if provided
                    if mask_latents is not None and mask_latents.shape[1] == x.shape[1]:
                        mask = mask_latents.squeeze(-1).bool()  # shape: [1, N] -> bool mask
                        update = torch.zeros_like(result)

                        masked_input = update_input[mask]  # shape: [N_masked, D]
                        masked_output = lora_B(lora_A(masked_input))
                        update[mask] = masked_output * scaling

                        result = result + update
                    else:
                        update = lora_B(lora_A(update_input)) * scaling
                        result = result + update
                else:
                    base_result = result if isinstance(dropout, torch.nn.Identity) or not self.training else None
                    if base_result is None:
                        x = dropout(x)

                    update = self.lora_magnitude_vector[active_adapter](
                        x,
                        lora_A=lora_A,
                        lora_B=lora_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                        base_result=base_result,
                    )
                    result = result + update

            result = result.to(torch_result_dtype)

        return result

    return new_forward


# Load models
unet = UNet2DConditionModel.from_pretrained(
    base_model_path, subfolder="unet", revision=False, variant=None
)
unet = unet.half()
unet.requires_grad_(False)
unet_lora_config = LoraConfig(
    use_dora=True,  # TODO
    r=4,
    lora_alpha=4,
    init_lora_weights="gaussian",
    target_modules=[
        "to_k", "to_q", "to_v", "to_out.0",
        "add_k_proj", "add_q_proj", "add_v_proj", "to_add_out"
    ],
)
unet.add_adapter(unet_lora_config)
unet = unet.to("cuda")

for name, module in unet.named_modules():
    if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
        module.forward = MethodType(make_new_forward(name), module)

brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionXLBrushNetPipeline.from_pretrained(
    base_model_path, brushnet=brushnet, torch_dtype=torch.float16, low_cpu_mem_usage=False,
    use_safetensors=True
).to("cuda")

pipe.vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
    
pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="style")
pipe.set_adapters("style")

def process_image(caption, output_filename):
    """Process image generation given a text caption and save the result."""
    init_image = cv2.imread(image_path)[:, :, ::-1]
    mask_image = 1. * (cv2.imread(mask_path).sum(-1) > 255)

    # Resize image
    h, w, _ = init_image.shape
    scale = 1024 / w if w < h else 1024 / h
    new_h, new_w = int(h * scale), int(w * scale)
    init_image = cv2.resize(init_image, (new_w, new_h))
    mask_image = cv2.resize(mask_image, (new_w, new_h))[:, :, np.newaxis]  # [new_h, new_w, 1]

    init_image = init_image * (1 - mask_image)
    init_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB")
    mask_image = Image.fromarray(mask_image.astype(np.uint8).repeat(3, -1) * 255).convert("RGB")  # [H, W, 3], 0/255, PIL

    lora_scale = 0.66
    generator = torch.Generator("cuda").manual_seed(333)
    image = pipe(
        prompt=caption,
        image=init_image,
        mask=mask_image,
        num_inference_steps=50,
        generator=generator,
        brushnet_conditioning_scale=brushnet_conditioning_scale,
        cross_attention_kwargs={"scale": lora_scale}
    ).images[0]

    image.save(output_filename)


# Read all .txt files in the folder
for txt_file in os.listdir(prompt_dir):
    if txt_file.endswith(".txt"):
        txt_path = os.path.join(prompt_dir, txt_file)
        with open(txt_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()
        output_filename = os.path.join(output_dir, f"{os.path.splitext(txt_file)[0]}.png")
        process_image(caption, output_filename)
        print(f"Generated image saved: {output_filename}")
