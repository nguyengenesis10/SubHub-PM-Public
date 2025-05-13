# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.


import os
import random
import dataclasses


from collections import Counter
from warnings import warn


import fire
import torch
import types
import numpy as np
import torch.optim as optim


from accelerate.utils import is_xpu_available


from llama_recipes.configs import (
    fsdp_config as FSDP_CONFIG,
    quantization_config as QUANTIZATION_CONFIG,
    train_config as TRAIN_CONFIG,
)
from llama_recipes.data.concatenator import ConcatDataset
from llama_recipes.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing

from llama_recipes.utils import fsdp_auto_wrap_policy
from llama_recipes.utils.config_utils import (
    check_fsdp_config,
    generate_dataset_config,
    generate_peft_config,
    get_dataloader_kwargs,
    update_config,
)
from llama_recipes.utils.dataset_utils import (
    get_custom_data_collator,
    get_preprocessed_dataset,
)

from llama_recipes.utils.fsdp_utils import hsdp_device_mesh
from llama_recipes.utils.train_utils import (
    clear_gpu_cache,
    freeze_transformer_layers,  # mllama specific
    freeze_LLM_only, # mllama specific
    get_policies,
    print_model_size, # compatible w/ any model
    print_frozen_model_status,  # compatible w/ any model
    setup,
    setup_environ_flags,
    train,
)

from peft import (
    get_peft_model, 
    PeftModel
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.optim.lr_scheduler import StepLR
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer
)
'''
from transformers.models.mllama.modeling_mllama import (
    MllamaCrossAttentionDecoderLayer,
    MllamaSelfAttentionDecoderLayer,
    MllamaVisionEncoderLayer,
)


from torch.nn.modules.conv import Conv2d
from torch.nn.modules.normalization import LayerNorm


from backend.modeling_mllama_vision import (
    MllamaPrecomputedAspectRatioEmbedding,
    MllamaPrecomputedPositionEmbedding,
    MllamaVisionMLP,
)
'''
from training import (
    count_trainable_params,
    freeze_model,
)
'''
from training.sft_utils import (
    prepare_vision_embedding_layers,
)
'''




# Setup weights & biases experiment tracking.
def setup_wandb(train_config, fsdp_config, **kwargs):
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "You are trying to use wandb which is not currently installed. "
            "Please install it using pip install wandb"
        )
    from llama_recipes.configs import wandb_config as WANDB_CONFIG

    wandb_config = WANDB_CONFIG()
    update_config(wandb_config, **kwargs)
    init_dict = dataclasses.asdict(wandb_config)
    run = wandb.init(**init_dict)
    run.config.update(train_config)
    run.config.update(fsdp_config, allow_val_change=True)
    return run


# Setup quantization config, currently supports only 4-bit w/ FSDP.
def setup_quant_config(train_config:TRAIN_CONFIG,**kwargs): 
    bnb_config=None
    if train_config.quantization:
        if type(train_config.quantization) == type(True):
            warn(
                "Quantization (--quantization) is a boolean, please specify quantization as '4bit' or '8bit'. Defaulting to '8bit' but this might change in the future.",
                FutureWarning
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


'''
Set up all configurations includes:
    train_config
    fsdp_config
    peft_config
    bnb_config
    dataset_config
    wandb_run
'''
class ConfigurationHandler:
    def __init__(self,**kwargs):
        train_config= TRAIN_CONFIG()
        fsdp_config=FSDP_CONFIG()
        update_config(
            (train_config, fsdp_config), **kwargs
        )
        self.train_config=train_config
        self.fsdp_config=fsdp_config
        self.bnb_config=setup_quant_config(
            self.train_config,**kwargs
        )
        self.peft_config=generate_peft_config(train_config, kwargs)
        self.dataset_config=generate_dataset_config(train_config,kwargs)
        
        wandb_run=None
        if train_config.use_wandb:
            if not train_config.enable_fsdp or rank == 0:
                wandb_run = setup_wandb(train_config, fsdp_config, **kwargs)
        self.wandb_run=wandb_run


class SFTTrainer(ConfigurationHandler):
    def __init__(
        self,
        **kwargs
    ):
        
        super().__init__(**kwargs)
        
        # Torchrun specific, for GPU assignment, downstream. 
        if self.train_config.enable_fsdp:
            setup()
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.rank = int(os.environ["RANK"])
        
        
        # Set the seeds for reproducibility
        if is_xpu_available():
            torch.xpu.manual_seed(train_config.seed)
            torch.manual_seed(train_config.seed)
            random.seed(train_config.seed)
            np.random.seed(train_config.seed)
    
    
    def setup_hsdp_device_mesh(self):
        hsdp_device_mesh_plan = None
        if (
            self.fsdp_config.hsdp and
            self.fsdp_config.sharding_strategy == ShardingStrategy.HYBRID_SHARD
        ):
            hsdp_device_mesh_plan = hsdp_device_mesh(
                replica_group_size=self.fsdp_config.replica_group_size,
                sharding_group_size=self.fsdp_config.sharding_group_size,
            )
            print("HSDP device mesh is ready")
        return hsdp_device_mesh_plan
    
    
    def train(self):
    
        if torch.distributed.is_initialized():
            if is_xpu_available():
                torch.xpu.set_device(self.local_rank)
            elif torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
            clear_gpu_cache(self.local_rank)
            setup_environ_flags(self.rank)
        
        model_initializer=ModelInitializer(
            self.train_config,self.bnb_config,self.fsdp_config,self.peft_config,self.wandb_run
        )
        model_initializer.initialize(rank=self.rank)
        model_initializer.setup_peft()
        model_initializer.add_full_finetuning_layers(
            rank=self.rank, update_func=self.train_config.configure_finetuning_layers
        )
        
        # Setup Hybrid Sharding device grid, if applicable.
        hsdp_device_mesh_plan=self.setup_hsdp_device_mesh()
        
        model_initializer.wrap_fsdp_model(
            rank=self.rank, hsdp_device_mesh_plan=hsdp_device_mesh_plan
        )
        
        model=model_initializer.model
        processor=model_initializer.processor
        tokenizer=model_initializer.tokenizer
        
        # Sanity check, only print on gpu0. 
        count_trainable_params(
            model,outputs=False,to_print=False
        ) if self.local_rank==0 else None
        
        train_dataloader,eval_dataloader=DataManager(
            dataset_config=self.dataset_config,train_config=self.train_config,processor=processor
        ).create_dataloaders()
        
        
        if self.fsdp_config.pure_bf16 and self.fsdp_config.optimizer == "anyprecision":
            optimizer = AnyPrecisionAdamW(
                model.parameters(),
                lr=self.train_config.lr,
                momentum_dtype=torch.bfloat16,
                variance_dtype=torch.bfloat16,
                use_kahan_summation=False,
                weight_decay=self.train_config.weight_decay,
            )
        else:
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.train_config.lr,
                weight_decay=self.train_config.weight_decay,
            )
        scheduler = StepLR(
            optimizer, step_size=1, gamma=self.train_config.gamma
        )
        results = train(
            model,
            train_dataloader,
            eval_dataloader,
            tokenizer,
            optimizer,
            scheduler,
            self.train_config.gradient_accumulation_steps,
            self.train_config,
            self.fsdp_config if self.train_config.enable_fsdp else None,
            self.local_rank if self.train_config.enable_fsdp else None,
            self.rank if self.train_config.enable_fsdp else None,
            self.wandb_run,
        )
        if not self.train_config.enable_fsdp or self.rank == 0:
            [print(f"Key: {k}, Value: {v}") for k, v in results.items()]
            if self.train_config.use_wandb:
                for k, v in results.items():
                    self.wandb_run.summary[k] = v


# Handles model, processor, and tokenizer instantiation, applies quantization, FSDP, etc. 
class ModelInitializer:
    def __init__(
        self,
        train_config:TRAIN_CONFIG,
        bnb_config:QUANTIZATION_CONFIG,
        fsdp_config:FSDP_CONFIG,
        peft_config,
        wandb_run,
    ):
        self.train_config=train_config
        self.bnb_config=bnb_config
        self.fsdp_config=fsdp_config
        self.peft_config=peft_config
        self.wandb_run=wandb_run
    
    
    # First initialization.
    def initialize(self,rank):
        
        config = AutoConfig.from_pretrained(
            self.train_config.model_name
        )
        if self.train_config.model_init_class:
            model_init_class=self.train_config.model_init_class
            model=model_init_class.from_pretrained(
                self.train_config.model_name,
                quantization_config=self.bnb_config,
                # use_cache= False if self.train_config.enable_fsdp else None,
                device_map=(
                    "auto" if self.train_config.quantization and not self.train_config.enable_fsdp else None
                ),
                torch_dtype=torch.float16 if self.train_config.use_fp16 else torch.bfloat16,
            )
            model.supports_gradient_checkpointing = True
        
        processor = AutoProcessor.from_pretrained(
            self.train_config.model_name if self.train_config.tokenizer_name is None else self.train_config.tokenizer_name
        )
        processor.image_processor.size = {
            'height': config.vision_config.image_size,
            'width': config.vision_config.image_size
        }
        processor.image_processor.max_image_tiles = config.vision_config.max_num_tiles
        processor.image_processor.patch_size = config.vision_config.patch_size
        
        # Load the tokenizer and add special tokens
        tokenizer = AutoTokenizer.from_pretrained(
            self.train_config.model_name if self.train_config.tokenizer_name is None else self.train_config.tokenizer_name
        )
        # If there's a size mismatch between the input space and the tokenizer, raise ValueError.
        if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
            raise ValueError(
                f"Shape mismatch, tokenizer has {len(tokenizer)} and model has embed shape {model.get_input_embeddings().weight.shape[0]}!"
            )
        # Make sure the tokenizer has an assigned pad token if not, automatically assign then end of sequence token.
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        print_model_size(
            model, self.train_config, rank if self.train_config.enable_fsdp else 0
        )
        if (
            self.train_config.enable_fsdp and 
            self.fsdp_config.pure_bf16 and not 
            self.train_config.quantization
        ):
            model.to(torch.bfloat16)
        
        self.model=model
        self.processor=processor
        self.tokenizer=tokenizer
        
        return (
            self.model,self.processor,self.tokenizer
        )
    
    
    # Setup PeFT for the model, currently only supports LoRA for LMMs.
    def setup_peft(self):
        if self.train_config.use_peft:
            # Load the pre-trained peft model checkpoint and setup its configuration
            if self.train_config.from_peft_checkpoint:
                self.model = PeftModel.from_pretrained(
                    self.model, self.train_config.from_peft_checkpoint, is_trainable=True
                )
                peft_config = self.model.peft_config
            # Generate the peft config and start fine-tuning from original model
            else:
                self.model = get_peft_model(self.model, self.peft_config)
            if wandb_run:
                wandb_run.config.update(self.peft_config)
            print(
                "\nModel utilizing PeFT."
            )
            self.model.print_trainable_parameters()
        else:
            print("\nModel not utilizing PeFT.")
    
    
    '''
    This function is meant to allow users to feed an extra set of layers, respective to that
    model for full parameter finetuning. For now, since my goal is to modularize the finetuning 
    script it'll just target the embedding layers. 
    
    You'll need to use a custom function that modifies your model in place. 
    
    This function acts as a wrapper to execute other functions. 
    '''
    def add_full_finetuning_layers(self, rank, update_func:list[types.FunctionType]):
        
        '''
        Llama-cookbook uses the below code block for layer targeting before wrapping w/ FSDP. Because, we're 
        trying to refactor the code base to rely on calling functions in the training configuration for layer 
        targeting I'm commenting it out and keeping it for reference.
        
        if (
            not self.train_config.use_peft 
            and self.train_config.freeze_layers
        ):
            freeze_transformer_layers(self.model, self.train_config.num_freeze_layers)
            # print model size and frozen layers after freezing layers
            print_frozen_model_status(
                self.model, self.train_config, rank if self.train_config.enable_fsdp else 0
            )
            
        if (
            not self.train_config.use_peft 
            and self.train_config.freeze_LLM_only 
            and self.model.config.model_type == "mllama"
        ):
            freeze_LLM_only(self.model)
            # print model size and frozen layers after freezing layers
            print_frozen_model_status(
                self.model, self.train_config, rank if self.train_config.enable_fsdp else 0
            )
        '''
        
        freeze_model(self.model)
        if isinstance(update_func,list):
            for func in update_func:
                func(self.model)
        elif isinstance(update_func,types.FunctionType):
            update_func(self.model)
        else:
            raise TypeError(
                f"'update_func' is suppose to be lists of functions or a function, but is {type(update_func)}!"
            )
        
    
    
    '''
    Need to go back in and confirm which one of the custom print functions require explicitly 
    models that are from the llama family.
    '''
    def wrap_fsdp_model(self,rank,hsdp_device_mesh_plan):
        if self.train_config.enable_fsdp:
            check_fsdp_config(
                self.fsdp_config
            )
            
            '''
            Default layers for sharding. In the llama-cookbook, they default to these layers unless 
            we're using PeFT. I've yet to validated differences in compute efficiency by just force 
            targeting the Mllama layers when we go to do SFT or in the case of knowledge distillation 
            we literally target all the vision layers see 'finetuning_distill.py' to see what I'm 
            talking abt. 
            
            transformer_layer_cls=set([
                LlamaDecoderLayer, 
                MllamaSelfAttentionDecoderLayer,
                MllamaVisionEncoderLayer,
                MllamaCrossAttentionDecoderLayer
            ])
            mixed_precision_policy, wrapping_policy = get_policies(self.fsdp_config, rank)
            
            B/c of this I just create my own wrapping policy on the spot, in the future we're going to 
            refactor so that wrapped layers is customizable based on usage and model. 
            '''
            mixed_precision_policy,_ = get_policies(self.fsdp_config, rank)
            if self.fsdp_config.wrapped_layers:
                if (
                    isinstance(self.fsdp_config.wrapped_layers,list) or 
                    isinstance(self.fsdp_config.wrapped_layers,type)
                ):
                    wrapped_layers = self.fsdp_config.wrapped_layers
                else:
                    raise TypeError(
                        f"'fsdp_config.wrapped_layers' is of type: {type(fsdp_config.wrapped_layers)}, should be a layer or a lists of layers to target!"
                    )
            else:
                raise ValueError(
                    f"'fsdp_config.wrapped_layers' is not defined, should be a layer or lists of layers to target!"
                )
            my_auto_wrapping_policy = fsdp_auto_wrap_policy(
                self.model,
                wrapped_layers
            )
            device_id = 0
            if is_xpu_available():
                device_id = torch.xpu.current_device()
            elif torch.cuda.is_available():
                device_id = torch.cuda.current_device()
            if self.train_config.freeze_LLM_only:
                use_orig_params = True
            else:
                use_orig_params = False
            
            # Instantiate the sharded and wrapped model.
            self.model = FSDP(
                self.model,
                auto_wrap_policy=my_auto_wrapping_policy,
                cpu_offload=(
                    CPUOffload(offload_params=True) if self.fsdp_config.fsdp_cpu_offload else None
                ),
                mixed_precision=(
                    mixed_precision_policy if not self.fsdp_config.pure_bf16 else None
                ),
                sharding_strategy=self.fsdp_config.sharding_strategy,
                device_mesh=hsdp_device_mesh_plan,
                device_id=device_id,
                limit_all_gathers=True,
                sync_module_states=self.train_config.low_cpu_fsdp,
                param_init_fn=(
                        (
                            lambda module: module.to_empty(device=torch.device("cuda"), 
                            recurse=False
                        )
                    )if self.train_config.low_cpu_fsdp and rank != 0 else None
                ),
                use_orig_params=use_orig_params,
            )
            if self.fsdp_config.fsdp_activation_checkpointing:
                self.model.enable_input_require_grads()
                self.model.gradient_checkpointing_enable()
                apply_fsdp_checkpointing(self.model)
        elif not self.train_config.quantization and not self.train_config.enable_fsdp:
            if is_xpu_available():
                self.model.to("xpu:0")
            elif torch.cuda.is_available():
                self.model.to("cuda")


# Creates data-loaders for training.
class DataManager:
    def __init__(
        self,
        dataset_config,
        train_config,
        processor,
    ):
        self.dataset_config=dataset_config
        self.train_config=train_config
        self.processor=processor
    
    
    '''
    All of SubHub's dataset in our repo have 2 splits a train and a test.
    We'll eventually need to refactor to handle N number of splits and processors.
    '''
    def fetch_datasets(self, splits:list[str]=["train","test"]):
        return tuple(
            get_preprocessed_dataset(
                self.processor,self.dataset_config,split 
            ) for split in splits
        )
    
    
    def create_dataloaders(self):
        
        dataset_train,dataset_val=self.fetch_datasets()
        
        if self.train_config.batching_strategy != "padding":
            raise ValueError(
                f"Must use batching strategy 'padding' with multimodal training not {self.train_config.batching_strategy}!"
            )
         
        train_dl_kwargs,val_dl_kwargs=tuple(
            get_dataloader_kwargs(
                self.train_config, dataset, self.processor, split
            ) for dataset,split in zip(
                [dataset_train,dataset_val] , ["train","test"]
            )
        )
        
        custom_data_collator = get_custom_data_collator(self.processor, self.dataset_config)
        if custom_data_collator:
            print(
                "\ncustom_data_collator is used."
            )
            train_dl_kwargs["collate_fn"] = custom_data_collator
        
        train_dataloader = torch.utils.data.DataLoader(
            dataset_train,
            num_workers=self.train_config.num_workers_dataloader,
            pin_memory=True,
            **train_dl_kwargs,
        )
        print(f"--> Num of Training Set Batches loaded = {len(train_dataloader)}")
        eval_dataloader=None
        if self.train_config.run_validation:
            val_dl_kwargs = get_dataloader_kwargs(
                self.train_config, dataset_val, self.processor, "val"
            )
            if custom_data_collator:
                val_dl_kwargs["collate_fn"] = custom_data_collator
            eval_dataloader = torch.utils.data.DataLoader(
                dataset_val,
                num_workers=self.train_config.num_workers_dataloader,
                pin_memory=True,
                **val_dl_kwargs,
            )
            if len(eval_dataloader) == 0:
                raise ValueError(
                    f"The eval set size is too small for dataloader to load even one batch. Please increase the size of eval set. ({len(eval_dataloader)=})"
                )
            else:
                print(f"--> Num of Validation Set Batches loaded = {len(eval_dataloader)}")
                
        return train_dataloader,eval_dataloader


if __name__ == "__main__":
    None







#