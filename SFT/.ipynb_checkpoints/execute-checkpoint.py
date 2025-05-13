
import os 
from datetime import datetime
from finetuning import (
    SFTTrainer,
    AutoConfig,
)
from training import (
    get_LoRA_target_modules,
    LoRA_target_modules_handler,
)
from training.sft_utils import (
    prepare_vision_embedding_layers,
    prepare_vision_layers,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy
)
from backend.modeling_mllama_vision import (
    MllamaProjectManagerForConditionalGeneration
)
from llama_recipes.utils.train_utils import (
    freeze_LLM_only
)
from transformers.models.mllama.modeling_mllama import (
    MllamaCrossAttentionDecoderLayer,
    MllamaSelfAttentionDecoderLayer,
    MllamaVisionEncoderLayer,
)
from transformers import MllamaForConditionalGeneration


model_name = "meta-llama/Llama-3.2-90B-Vision-Instruct"
# model_name = "SubHub/mllama3.3_Acronym_X_Text_I1200_T25_P75"
tokenizer_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
general_kwargs={
    "run_validation": True,
    "lr": 1e-5,
    "num_epochs": 1,
    "batch_size_training": 1,
    "model_name": model_name,
    "model_init_class" : MllamaForConditionalGeneration,
    "configure_finetuning_layers" : [
        prepare_vision_embedding_layers,
        prepare_vision_layers,
        # freeze_LLM_only,
    ],
    "train_config.tokenizer_name" : tokenizer_name,
    # Forces 'orig_params=True' when the model gets wrapped by FSDP allowing custom layers to be 
    # on vs. off instead of homogeneity between wrapped layer types. FSDP essentially 'requires_grad=True' 
    # for all submodules within a layer by 'freeze_LLM-only=True' then forced 'orig_params=True'. 
    # If training, vision_embeddings_only has to be set to True regardless of PeFT as it
    # freezes self-attention-decoder layers, but keeps cross + vision-encoder layers 'requires_grad=True'.
    "train_config.freeze_LLM_only" : True, # if vision_embeddings_only else False,
    # "train_config.gradient_accumulation_steps" : int,  # number of gradients to gather before performing back-prop
    # "train_config.gradient_clipping" : bool,  # determine whether of not to clip a gradient if it hits a certain threshold, prevent exploding gradients
    # "train_config.gradient_clipping_threshold" : int, 
    
    "use_fast_kernels": True,
}


fsdp_config_hsdp = False
fsdp_kwargs={
    "enable_fsdp": True,
    "fsdp_config.hsdp" : fsdp_config_hsdp,
    # if hsdp==False  
    # Option between ShardingStrategy.FULL_SHARD and ShardingStrategy.HYBRID_SHARD
    "fsdp_config.sharding_strategy" : ShardingStrategy.HYBRID_SHARD if fsdp_config_hsdp else ShardingStrategy.FULL_SHARD, 
    "fsdp_config.sharding_group_size" : 8,
    "fsdp_config.replica_group_size" : 1,
    "fsdp_config.wrapped_layers" : [
        MllamaSelfAttentionDecoderLayer,
        MllamaCrossAttentionDecoderLayer,
        MllamaVisionEncoderLayer,
    ],
}


root=os.path.expanduser("~")
save_kwargs={
    # both are used if enable_fsdp==True
    "dist_checkpoint_root_folder": f"{root}/save_model/",
    "dist_checkpoint_folder" : "checkpoints",
    
    # Saves training analytics into a json file
    "output_dir" : f"{root}/save_model/PEFT",
    "save_metrics" : True,
    
    # wandb login
    # 042a7cfe5ea36fac413ac8d335f9715d3402d798
    "use_wandb" : False,
    "wandb_config.project" : f"{datetime.now().year}{datetime.now().month:02d}{datetime.now().day}_mllama3.2-11b_1st_test",
    
    # Enable flop counter to measure model throughput, can not be used with pytorch profiler at the same time.
    "flop_counter" : True, 
    "flop_counter_start" : 3, 
}

use_peft = False # if set to True uses LoRA to optimize training
if use_peft:
    peft_kwargs={
        # PEFT config - only LoRA is avaliable
        "use_peft": use_peft,
        "peft_method": "lora",
        "lora_config.target_modules" : LoRA_target_modules_handler(funcs=get_LoRA_target_modules)
    }
else:
    peft_kwargs={}


dataset_kwargs={
    "batching_strategy": "padding",
    "dataset": "custom_dataset",
    "custom_dataset.test_split": "test",
    "custom_dataset.file": "datasets/pg16_WPAFB_dataset_mllama.py",
    "train_config.num_workers_dataloader" : 1,
}


'''
torchrun --nnodes 1 --nproc_per_node 8 --node_rank=0 execute.py
'''
sft_trainer=SFTTrainer(
    **{
        **general_kwargs,
        **fsdp_kwargs,
        **save_kwargs,
        **peft_kwargs,
        **dataset_kwargs,
    }
)
sft_trainer.train()




#