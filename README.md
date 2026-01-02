## 🧚‍♀️ FairyGen: Storied Cartoon Video from a Single Child-Drawn Character

<b>[Jiayi Zheng]() and [Xiaodong Cun](http://vinthony.github.io)</b> from [GVC Lab @ Great Bay University](http://gvclab.github.io)

[Arxiv](https://arxiv.org/abs/2506.21272) | [PDF](https://arxiv.org/pdf/2506.21272) | [Project Page](https://jayleejia.github.io/FairyGen/)

---

TL;DR: We create story videos from a single child-drawn character image.

![robot](assets/robot_adventure.gif)


### Main Pipeline 

![pipeline](assets/pipeline.png)

We propose FairyGen, a novel framework for generating animated story videos from a single hand-drawn character, while faithfully preserving its artistic style. It features story planning via MLLM, propagated stylization, 3D-based motion generation, and a two-stage propagated motion adapter.

### Getting Started

FairyGen is a dual-pipeline framework designed for high-fidelity character stylization using **SDXL** and motion-consistent animation using **Wan2.2**.

### 🚀 Environment Setup

```bash
conda create -n fairygen python=3.12 -y
conda activate fairygen

pip install -r requirements.txt

# Install modified diffusers for BrushNet support
cd stylization/BrushNet
pip install -e .
```

**[NOTE] Diffusers Library Compatibility:** BrushNet requires the modified version of `diffusers` library to support specific code updates. While standard Style LoRA/DoRA or two-stage animation finetuning can operate on the latest official `diffusers` release, the local installation (`pip install -e .`) is mandatory for BrushNet functionality.

### 📦 Model Download

**For Stylization (SDXL & BrushNet):**

```bash
hf download stabilityai/stable-diffusion-xl-base-1.0  
hf download madebyollin/sdxl-vae-fp16-fix 
```

BrushNet Checkpoint: available at this [Google Drive Link](https://drive.google.com/drive/folders/1KBr71RlQEACJPcs2Uoanpi919nISpG1L)

**For Animation (Wan2.2-TI2V-5B):**

```bash
hf download Wan-AI/Wan2.2-TI2V-5B
```

**[NOTE] Local Model Loading:** To load weights from local directories, modify the `--model_id_with_origin_paths` in `stage1_id.sh` to `model_paths`, and load the models in JSON format as shown below:

```bash
--model_paths '[
    [
      ".models/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003-bf16.safetensors",
      ".models/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003-bf16.safetensors",
      "./models/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003-bf16.safetensors"
    ],
    "./models/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth",
    "./models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
]'
```

### 🎨 Stylization

Step 1: Style DoRA Training

The goal is to learn the character's visual style. The training data requires a single character image paired with its binary mask. A script to generate binary masks is provided ([create_mask.py](stylization/dora_training/create_mask.py)). Example datasets can be found in [here](stylization/dora_training/data/train).

```bash
cd stylization/dora_training
bash train.sh
```

Step 2: Background Generation

After training the style DoRA, BrushNet is used for generating consistent backgrounds. Example data is available [here](./stylization/BrushNet/data/). When crafting prompts, it is recommended to include a description of the character’s appearance for better results.

A key parameter when using BrushNet is `brushnet_conditioning_scale`, which controls the trade-off between style consistency and background richness. Higher values (e.g., 1.0) emphasize style consistency, while lower values allow for more text alignment and richer background content. A value of `0.7` is commonly used.

```bash
cd stylization/BrushNet
python examples/brushnet/test_brushnet_sdxl.py
```

### 🎬 Animation

A two-stage training approach is applied to learn anthropomorphic motion. Example dataset can be found [here](animation/data/pig_walk).

Stage1: Learn character identity (appearance)

```bash
cd animation
bash stage1_id.sh
```

Stage 2: Learn motion information. Update `lora_checkpoint` with the checkpoint from stage 1 before stage 2 training. More complex motions may require additional training steps in stage 2.

```bash
bash stage2_motion.sh
```

Merge Two-Stage LoRA

```bash
python merge_weights.py
```

**Generate Animation**: After merging the two-stage LoRA, generate animation by using the first frame with a background and loading the merged motion LoRA. 

```bash
# single shot
python inference.py

# multi-shot 
python batch_inference.py
```


### Citation

```bibtex

@article{zheng2025fairygen,
      title={FairyGen: Storied Cartoon Video from a Single Child-Drawn Character}, 
      author={Jiayi Zheng and Xiaodong Cun},
      year={2025},
      eprint={2506.21272},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2506.21272}, 
}

```