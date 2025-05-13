
import sys ; sys.path.append("/home/ubuntu/SubHub-South-TX") ; sys.path.append("/home/ubuntu")

#### ---------Llama-Recipes-----------

from llama_recipes.utils.fsdp_utils import hsdp_device_mesh
from llama_recipes.data.concatenator import ConcatDataset
from llama_recipes.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing
from llama_recipes.utils import fsdp_auto_wrap_policy


from llama_recipes.configs import (
    fsdp_config as FSDP_CONFIG,
    quantization_config as QUANTIZATION_CONFIG,
    train_config as TRAIN_CONFIG,
)

from llama_recipes.utils.train_utils import (
    train,
    freeze_transformer_layers,
    setup,
    cleanup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies,
)
from llama_recipes.utils.config_utils import (
    update_config,
    generate_peft_config,
    get_dataloader_kwargs,
)

#### ---------TORCH Modules-----------

import torch
import torch.optim as optim

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy
)
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.optim.lr_scheduler import StepLR

#### -----------Transformers-----------

# for text model
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
# for multi-modal model
from transformers.models.mllama.modeling_mllama import (
    MllamaCrossAttentionDecoderLayer,
    MllamaSelfAttentionDecoderLayer
)

#### --------------CustomMllamaVision--------------

# since we've updated vision model on a different package in order for type-checking to be effective 
# these layers are compute intensive so we're going to shard them
from backend.modeling_mllama_vision import (
    MllamaVisionEncoderLayer,
    MllamaVisionFeatureCrossAttention,
    MllamaPrecomputedPositionEmbedding,
    MllamaPrecomputedAspectRatioEmbedding
)

#### --------------IMPORTS--------------

from accelerate.utils import is_xpu_available
from peft import get_peft_model, PeftModel 

from collections import Counter
import os

import dataclasses
import fire
import random




def get_bnb_config(**kwargs):
    # setting quantization configs
    train_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()
    update_config( (train_config, fsdp_config) , **kwargs )
    
    bnb_config = None
    if train_config.quantization:
        if type(train_config.quantization) == type(True):
            warn(
                "Quantization (--quantization) is a boolean, please specify quantization as '4bit' or '8bit'. Defaulting to '8bit' but this might change in the future.",
                FutureWarning,
            )
            train_config.quantization = "8bit"

        if train_config.quantization == "8bit" and train_config.enable_fsdp:
            raise ValueError(
                "8bit quantization is not supported with FSDP, please use 4bit quantization"
            )

        quant_config = QUANTIZATION_CONFIG()
        update_config(quant_config, **kwargs)
        bnb_config = quant_config.create_bnb_config(train_config.quantization)

        return bnb_config

def homogenize_FSDP_modules(sub_models, allowed_layers):
    """
    Ensures all parameters in allowed layers across all submodels have requires_grad=True.

    Args:
        submodels (list): A list of submodels to process.
        allowed_layers (tuple): A tuple of layer types to check.
    """
    def process_module(module):
        # Check all submodules recursively
        for name, child in module.named_children():
            # convert to tuple if 'allowed_layers' is a list 
            if isinstance(child, tuple(allowed_layers)):
                # Update requires_grad for parameters in the allowed layer
                for param_name, param in child.named_parameters():
                    if not param.requires_grad:
                        param.requires_grad = True
            # Recursively process nested submodules
            process_module(child)

    # Process each submodel
    for sub_model in sub_models:
        process_module(sub_model)



#