import safetensors.torch
import torch
import os

stage1_path = "./lora-libraries/pig_walk/stage1/step-200.safetensors"
stage2_path = "./lora-libraries/pig_walk/stage2/step-400.safetensors"
merged_path = "./lora-libraries/pig_walk/merged/200_400.safetensors"
keys_log_path = None

os.makedirs(os.path.dirname(merged_path), exist_ok=True)

def get_inner_module(module):
    if hasattr(module, '_orig_mod'):
        module = module._orig_mod
    if hasattr(module, '_checkpoint_wrapped_module'):
        module = module._checkpoint_wrapped_module
    return module

def merge_lora_weights(stage1_path, stage2_path, save_path, keys_log_path=None):
    stage1 = safetensors.torch.load_file(stage1_path)
    stage2 = safetensors.torch.load_file(stage2_path)

    if keys_log_path is not None:
        with open(keys_log_path, "w", encoding="utf-8") as f:
            for key in stage2.keys():
                f.write(key + "\n")

    # B = B1 + B2
    merged_weights = {}
    for k in stage1.keys():
        if "lora_A" in k:
            merged_weights[k] = stage1[k]
        elif "lora_B" in k:
            if k.endswith(".lora_B.default.weight"):
                b2_key = k.replace(".lora_B.default.weight", ".lora_B2.weight")
            else:
                b2_key = k.replace("lora_B", "lora_B2").replace(".default", "")

            if b2_key in stage2:
                merged_weights[k] = stage1[k] + stage2[b2_key]
            else:
                print("Warning: Missing B2 key for", k, "→ 期望的 b2_key:", b2_key)
                merged_weights[k] = stage1[k]

    safetensors.torch.save_file(merged_weights, save_path)


merge_lora_weights(stage1_path, stage2_path, merged_path, keys_log_path)
