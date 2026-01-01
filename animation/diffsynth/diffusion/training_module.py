import torch, json
from ..core import ModelConfig, load_state_dict
from ..utils.controlnet import ControlNetInput
from peft import LoraConfig, inject_adapter_in_model


class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self
        
        
    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules
    
    
    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names
    
    
    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None, upcast_dtype=None):
        """ Add LoRA adapters to the model. """
        if lora_alpha is None:
            lora_alpha = lora_rank
        if isinstance(target_modules, list) and len(target_modules) == 1:
            target_modules = target_modules[0]
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            # use_dora=True,
            init_lora_weights=True
        )
        model = inject_adapter_in_model(lora_config, model)

        if upcast_dtype is not None:
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.to(upcast_dtype)
        return model


    def mapping_lora_state_dict(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if "lora_A.weight" in key or "lora_B.weight" in key:
                new_key = key.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
                new_state_dict[new_key] = value
            elif "lora_A.default.weight" in key or "lora_B.default.weight" in key:
                new_state_dict[key] = value
        return new_state_dict


    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict
    
    
    def transfer_data_to_device(self, data, device, torch_float_dtype=None):
        if data is None:
            return data
        elif isinstance(data, torch.Tensor):
            data = data.to(device)
            if torch_float_dtype is not None and data.dtype in [torch.float, torch.float16, torch.bfloat16]:
                data = data.to(torch_float_dtype)
            return data
        elif isinstance(data, tuple):
            data = tuple(self.transfer_data_to_device(x, device, torch_float_dtype) for x in data)
            return data
        elif isinstance(data, list):
            data = list(self.transfer_data_to_device(x, device, torch_float_dtype) for x in data)
            return data
        elif isinstance(data, dict):
            data = {i: self.transfer_data_to_device(data[i], device, torch_float_dtype) for i in data}
            return data
        else:
            return data
    
    def parse_vram_config(self, fp8=False, offload=False, device="cpu"):
        if fp8:
            return {
                "offload_dtype": torch.float8_e4m3fn,
                "offload_device": device,
                "onload_dtype": torch.float8_e4m3fn,
                "onload_device": device,
                "preparing_dtype": torch.float8_e4m3fn,
                "preparing_device": device,
                "computation_dtype": torch.bfloat16,
                "computation_device": device,
            }
        elif offload:
            return {
                "offload_dtype": "disk",
                "offload_device": "disk",
                "onload_dtype": "disk",
                "onload_device": "disk",
                "preparing_dtype": torch.bfloat16,
                "preparing_device": device,
                "computation_dtype": torch.bfloat16,
                "computation_device": device,
                "clear_parameters": True,
            }
        else:
            return {}
    
    def parse_model_configs(self, model_paths, model_id_with_origin_paths, fp8_models=None, offload_models=None, device="cpu"):
        fp8_models = [] if fp8_models is None else fp8_models.split(",")
        offload_models = [] if offload_models is None else offload_models.split(",")
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            for path in model_paths:
                vram_config = self.parse_vram_config(
                    fp8=path in fp8_models,
                    offload=path in offload_models,
                    device=device
                )
                model_configs.append(ModelConfig(path=path, **vram_config))
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            for model_id_with_origin_path in model_id_with_origin_paths:
                model_id, origin_file_pattern = model_id_with_origin_path.split(":")
                vram_config = self.parse_vram_config(
                    fp8=model_id_with_origin_path in fp8_models,
                    offload=model_id_with_origin_path in offload_models,
                    device=device
                )
                model_configs.append(ModelConfig(model_id=model_id, origin_file_pattern=origin_file_pattern, **vram_config))
        return model_configs
    
    
    def switch_pipe_to_training_mode(
        self,
        pipe,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        task="sft",
    ):
        # Scheduler
        pipe.scheduler.set_timesteps(1000, training=True)
        
        # Freeze untrainable models
        pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        
        # Preset LoRA
        if preset_lora_path is not None:
            pipe.load_lora(getattr(pipe, preset_lora_model), preset_lora_path)
        
        # FP8
        # FP8 relies on a model-specific memory management scheme.
        # It is delegated to the subclass.
        
        # Add LoRA to the base models
        if lora_base_model is not None and not task.endswith(":data_process"):
            if (not hasattr(pipe, lora_base_model)) or getattr(pipe, lora_base_model) is None:
                print(f"No {lora_base_model} models in the pipeline. We cannot patch LoRA on the model. If this occurs during the data processing stage, it is normal.")
                return

            # print("getattr(pipe, lora_base_model):")
            # print(getattr(pipe, lora_base_model))  # WanModel

            # Add LoRA adapters
            model = self.add_lora_to_model(
                getattr(pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank,
                upcast_dtype=pipe.torch_dtype,
            )

            # Load LoRA checkpoint
            if lora_checkpoint is not None:
                state_dict = load_state_dict(lora_checkpoint)
                state_dict = self.mapping_lora_state_dict(state_dict)
                load_result = model.load_state_dict(state_dict, strict=False)
                print(f"LoRA checkpoint loaded: {lora_checkpoint}, total {len(state_dict)} keys")
                if len(load_result[1]) > 0:
                    print(f"Warning, LoRA key mismatch! Unexpected keys in LoRA checkpoint: {load_result[1]}")

            from types import MethodType
            from typing import Any
            import torch.nn.functional as F

            if lora_checkpoint is None:
                # stage-1 training
                print("stage-1 training")
                for name, module in model.named_modules():
                    if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                        def new_forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
                            self._check_forward_args(x, *args, **kwargs)
                            adapter_names = kwargs.pop("adapter_names", None)

                            if self.disable_adapters:
                                if self.merged:
                                    self.unmerge()
                                result = self.base_layer(x, *args, **kwargs)
                            elif adapter_names is not None:
                                result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
                            elif self.merged:
                                result = self.base_layer(x, *args, **kwargs)
                            else:
                                # execute this block
                                result = self.base_layer(x, *args, **kwargs)  # Wx
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
                                        dropout_prob = 0.8
                                        mask = (torch.rand_like(lora_B.weight) > dropout_prob).to(lora_B.weight.dtype).to(
                                            lora_B.weight.device)
                                        scale_factor = 1.0 / (1 - dropout_prob)
                                        B_dropped = lora_B.weight * mask * scale_factor

                                        intermediate = lora_A(dropout(x))
                                        update = F.linear(intermediate, B_dropped, lora_B.bias)
                                        result = result + update * scaling
                                    else:
                                        if isinstance(dropout, torch.nn.Identity) or not self.training:
                                            base_result = result
                                        else:
                                            x = dropout(x)
                                            base_result = None

                                        result = result + self.lora_magnitude_vector[active_adapter](
                                            x,
                                            lora_A=lora_A,
                                            lora_B=lora_B,
                                            scaling=scaling,
                                            base_layer=self.get_base_layer(),
                                            base_result=base_result,
                                        )

                                result = result.to(torch_result_dtype)

                            return result

                        # replace forward method
                        module.forward = MethodType(new_forward, module)

            else:
                print("stage-2 training")
                # stage-2 training
                # 1. Load stage-1 LoRA checkpoint (A1, B1)
                stage1_lora_weights = load_state_dict(lora_checkpoint)
                stage1_lora_weights = self.mapping_lora_state_dict(stage1_lora_weights)
                model.load_state_dict(stage1_lora_weights, strict=False)

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                upcast_dtype = pipe.torch_dtype
                # 2. Add additional LoRA B2 layers
                for name, module in model.named_modules():
                    if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                        # 1. freeze A1, B1
                        for lora_block in [module.lora_A, module.lora_B]:
                            if isinstance(lora_block, torch.nn.Module):
                                for p in lora_block.parameters():
                                    p.requires_grad = False
                            elif isinstance(lora_block, dict):  
                                for sub in lora_block.values():
                                    if isinstance(sub, torch.nn.Module):
                                        for p in sub.parameters():
                                            p.requires_grad = False

                        # Register a new B2 layer for additional LoRA update
                        lora_B2_layer = module.lora_B['default']
                        module.lora_B2 = torch.nn.Linear(
                            in_features=lora_B2_layer.weight.shape[1],
                            out_features=lora_B2_layer.weight.shape[0],
                            bias=False
                        ).to(device)

                        # Zero initialize B2
                        with torch.no_grad():
                            module.lora_B2.weight.zero_()

                        # Convert B2 dtype
                        if upcast_dtype is not None:
                            module.lora_B2.weight.data = module.lora_B2.weight.data.to(upcast_dtype)

                        # Enable B2 parameters for training
                        module.lora_B2.weight.requires_grad = True
                        
                        def dump_param_names(model, file_path="param_names.txt"):
                            with open(file_path, "w", encoding="utf-8") as f:
                                for name, param in model.named_parameters():
                                    line = f"{name}\tshape={tuple(param.shape)}\trequires_grad={param.requires_grad}\n"
                                    f.write(line)

                        dump_param_names(model, "param_names.txt")

                        def new_forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
                            self._check_forward_args(x, *args, **kwargs)

                            result = self.base_layer(x, *args, **kwargs)  # Wx
                            torch_result_dtype = result.dtype

                            lora_A_keys = self.lora_A.keys()
                            for active_adapter in self.active_adapters:
                                if active_adapter not in lora_A_keys:
                                    continue

                                lora_A = self.lora_A[active_adapter]
                                lora_B = self.lora_B[active_adapter]
                                lora_B2 = self.lora_B2

                                dropout = self.lora_dropout[active_adapter]
                                scaling = self.scaling[active_adapter]
                                x = x.to("cuda")

                                result = result + lora_B(lora_A(dropout(x))) * scaling  # Wx + A1B1x

                                dropout_prob = 0.5  
                                mask = (torch.rand_like(lora_B2.weight) > dropout_prob).to(lora_B2.weight.dtype).to(
                                    lora_B2.weight.device)
                                scale_factor = 1.0 / (1 - dropout_prob)
                                B2_dropped = lora_B2.weight * mask * scale_factor

                                intermediate = lora_A(dropout(x))
                                update = F.linear(intermediate, B2_dropped, lora_B.bias)
                                result = result + update * scaling

                                result = result.to(torch_result_dtype)

                            return result

                        module.forward = MethodType(new_forward, module)

            setattr(pipe, lora_base_model, model)


    def split_pipeline_units(self, task, pipe, trainable_models=None, lora_base_model=None):
        models_require_backward = []
        if trainable_models is not None:
            models_require_backward += trainable_models.split(",")
        if lora_base_model is not None:
            models_require_backward += [lora_base_model]
        if task.endswith(":data_process"):
            _, pipe.units = pipe.split_pipeline_units(models_require_backward)
        elif task.endswith(":train"):
            pipe.units, _ = pipe.split_pipeline_units(models_require_backward)
        return pipe
    
    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        controlnet_keys_map = (
            ("blockwise_controlnet_", "blockwise_controlnet_inputs",),
            ("controlnet_", "controlnet_inputs"),
        )
        controlnet_inputs = {}
        for extra_input in extra_inputs:
            for prefix, name in controlnet_keys_map:
                if extra_input.startswith(prefix):
                    if name not in controlnet_inputs:
                        controlnet_inputs[name] = {}
                    controlnet_inputs[name][extra_input.replace(prefix, "")] = data[extra_input]
                    break
            else:
                inputs_shared[extra_input] = data[extra_input]
        for name, params in controlnet_inputs.items():
            inputs_shared[name] = [ControlNetInput(**params)]
        return inputs_shared
