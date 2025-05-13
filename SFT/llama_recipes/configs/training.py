# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


class student_config:
    model_name: str=None


class teacher_config:
    model_name: str=None


@dataclass
class train_config:
    model_name: str="PATH/to/Model"
    model_init_class=None # class used to initialize model from
    configure_finetuning_layers=None # function used to specify which layers are going to be targeted for fill parameter finetuning
    tokenizer_name: str=None
    enable_fsdp: bool=False # shards model parameters, optimizer states and gradients across DDP ranks
    low_cpu_fsdp: bool=False # saves cpu memory by loading pretrained model on rank0 only
    run_validation: bool=True
    batch_size_training: int=4
    batching_strategy: str="packing" #alternative: padding
    context_length: int=4096
    gradient_accumulation_steps: int=1
    gradient_clipping: bool = False
    gradient_clipping_threshold: float = 1.0
    num_epochs: int=3
    max_train_step: int=0
    max_eval_step: int=0
    num_workers_dataloader: int=1
    lr: float=1e-4
    weight_decay: float=0.0
    gamma: float= 0.85 # multiplicatively decay the learning rate by gamma after each epoch
    seed: int=42
    use_fp16: bool=False
    mixed_precision: bool=True
    val_batch_size: int=1
    dataset = "samsum_dataset"
    peft_method: str = "lora" # None, llama_adapter (Caution: llama_adapter is currently not supported with FSDP)
    use_peft: bool=False # use parameter efficient fine tuning
    from_peft_checkpoint: str="" # if not empty and use_peft=True, will load the peft checkpoint and resume the fine-tuning on that checkpoint
    output_dir: str = "PATH/to/save/PEFT/model"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    freeze_LLM_only: bool = False # Freeze self-attention layers in the language_model. Vision model, multi_modal_projector, cross-attention will be fine-tuned
    quantization: str = None
    one_gpu: bool = False
    save_model: bool = True
    dist_checkpoint_root_folder: str="PATH/to/save/FSDP/model" # will be used if using FSDP
    dist_checkpoint_folder: str="fine-tuned" # will be used if using FSDP
    save_optimizer: bool=False # will be used if using FSDP
    use_fast_kernels: bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    use_wandb: bool = False # Enable wandb for experient tracking
    save_metrics: bool = False # saves training metrics to a json file for later plotting
    flop_counter: bool = False # Enable flop counter to measure model throughput, can not be used with pytorch profiler at the same time.
    flop_counter_start: int = 3 # The step to start profiling, default is 3, which means after 3 steps of warmup stage, the profiler will start to count flops.
    use_profiler: bool = False # Enable pytorch profiler, can not be used with flop counter at the same time.
    profiler_dir: str = "PATH/to/save/profiler/results" # will be used if using profiler


@dataclass
class distill_train_config(train_config):
    pass 


'''

( 88.44 / 24.78 ) / 8 

88.44 
----- x 8 / 210 = secs for sample 
24.78 


def compute_gpu_hrs_per_epoch(
    total_flop:int,
    flops:int,
    dataset_estimate:int=3_000_000,
    batch_size:int=1,
    dataset_size:int=210,
):
    import numpy as np 
    return np.round(
        ((((total_flop / flops) * 8 ) / dataset_size) * dataset_estimate) / (60 *60) , 2
    )

def compute_gpu_theo_hrs_per_epoch(
    total_flop:int,
    dataset_estimate:int=3_000_000,
    batch_size:int=1,
    dataset_size:int=210,
):
    import numpy as np 
    return np.round(
        ((((total_flop / 1_979_000) * 8 ) / dataset_size) * dataset_estimate) / (60 *60) , 5
    )


total_flop_list = [88.44,88.44,23.17,90.44,90.44,90.44]
flops_list = [24.78,25.34,16.23,74.91,62.00,63.00]
exp_gpu_hrs = []
theo_gpu_hrs = []
for total_flop, flops in zip(total_flop_list,flops_list):
    exp_gpu_hrs.append(
       compute_gpu_hrs_per_epoch(
           total_flop,
           flops
        )
    )
    theo_gpu_hrs.append(
        compute_gpu_theo_hrs_per_epoch(
            total_flop
        )
    )


exp_gpu_hrs
theo_gpu_hrs


'''