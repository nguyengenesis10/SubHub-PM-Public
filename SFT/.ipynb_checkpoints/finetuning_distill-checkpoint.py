# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.


import dataclasses
import os
import random
from collections import Counter
from warnings import warn

import fire
import numpy as np
import torch
import torch.optim as optim
from accelerate.utils import is_xpu_available
from typing import Dict

### ----------llama_recipes---------

from llama_recipes.configs import (
    distill_train_config as DISTILL_TRAIN_CONFIG,
    quantization_config as QUANTIZATION_CONFIG,
    student_config as STUDENT_CONFIG,
    teacher_config as TEACHER_CONFIG,
    train_config as TRAIN_CONFIG,
    fsdp_config as FSDP_CONFIG,
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
    freeze_transformer_layers,
    freeze_LLM_only,
    get_policies,
    print_model_size,
    print_frozen_model_status,
    setup,
    setup_environ_flags,
)

from llama_recipes.utils.distill_utils import (
    train,
)


### ----------llama_recipes---------

from peft import get_peft_model, PeftModel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.optim.lr_scheduler import StepLR
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaForCausalLM,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mllama.modeling_mllama import (
    MllamaCrossAttentionDecoderLayer,
    MllamaSelfAttentionDecoderLayer,
    MllamaVisionEncoderLayer as TeacherMllamaVisionEncoderLayer,
)
from backend.modeling_mllama_vision import (
    MllamaPrecomputedAspectRatioEmbedding,
    MllamaPrecomputedPositionEmbedding,
    MllamaVisionEncoderLayer as StudentMllamaVisionEncoderLayer,
)
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.normalization import LayerNorm

from training import (
    freeze_model,
    count_trainable_params,
)
from training.distill_utils import (
    prep_processor,
    check_processor,
    fetch_all_datasets,
    create_dataloaders,
    prepare_vision_embedding_layers,
    processor_init_dict,
    config_init_dict,
    model_init_dict,
)


# Setup Weights & Biases for experiment tracking.
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


# focusing on low-res distillation rn
def distill_main(
    subhub=False,
    vision_embeddings_only=False,
    no_language_model=False,
    student_processor_cfg:list[Dict] = None,
    **kwargs
):
    '''
    Args:
        subhub: Determines the model class to use for instantiation, if set to True uses 
                a SubHub custom model-class, else just classic MllamaForConditionalGeneration.
        vision_embeddings_only: If set to True, sets gradient updates only on the vision embedding layers 
                                and sets of a chain of custom logic which ensures all other layers are frozen
                                properly.
        no_language_model: If set to True, only vision heads are instantiated and sets of chain of custom logic
                           to sensure proper layer wrapping for FSDP.
        student_processor_cfg: Takes a dictionary that has keys ["image_size" , "max_num_tiles"] to modify the 
                               processor, allows us to gradually increase image resolution. 
    '''
    
    # Update the configuration for the training and sharding process
    distill_train_config = DISTILL_TRAIN_CONFIG()
    teacher_config = TEACHER_CONFIG() 
    student_config = STUDENT_CONFIG()
    fsdp_config = FSDP_CONFIG()
    update_config( 
        (
            distill_train_config,
            fsdp_config,
            teacher_config,
            student_config
        ), 
        **kwargs 
    )
    # Set the seeds for reproducibility
    if is_xpu_available():
        torch.xpu.manual_seed(distill_train_config.seed)
    torch.manual_seed(distill_train_config.seed)
    random.seed(distill_train_config.seed)
    np.random.seed(distill_train_config.seed)

    if distill_train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        if is_xpu_available():
            torch.xpu.set_device(local_rank)
        elif torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    wandb_run = None

    if distill_train_config.use_wandb:
        if not distill_train_config.enable_fsdp or rank == 0:
            wandb_run = setup_wandb(distill_train_config, fsdp_config, **kwargs)

    # setting quantization configs
    bnb_config = None
    if distill_train_config.quantization:
        if type(distill_train_config.quantization) == type(True):
            warn(
                "Quantization (--quantization) is a boolean, please specify quantization as '4bit' or '8bit'. Defaulting to '8bit' but this might change in the future.",
                FutureWarning,
            )
            distill_train_config.quantization = "8bit"

        if distill_train_config.quantization == "8bit" and distill_train_config.enable_fsdp:
            raise ValueError(
                "8bit quantization is not supported with FSDP, please use 4bit quantization"
            )

        quant_config = QUANTIZATION_CONFIG()
        update_config(quant_config, **kwargs)
        bnb_config = quant_config.create_bnb_config(distill_train_config.quantization)

    # Load the pre-trained model and setup its configuration
    use_cache = False if distill_train_config.enable_fsdp else None
    
    config_class_init = config_init_dict["vision"] if no_language_model else config_init_dict["multimodal"]
    
    student_model_config = config_class_init.from_pretrained(student_config.model_name)
    teacher_model_config = config_class_init.from_pretrained(teacher_config.model_name)
    
    if (
            (
            student_model_config.model_type == "mllama" and teacher_model_config.model_type == "mllama"
            )
        or no_language_model is not None
     ):
        is_vision = True

        # determine which class to use for model initialization
        if no_language_model:
            student_class_init = model_init_dict["student"]["vision"]
            teacher_class_init = model_init_dict["teacher"]["vision"]
        else:
            student_class_init = model_init_dict["student"]["multimodal"]
            teacher_class_init = model_init_dict["teacher"]["multimodal"]
        
        student = student_class_init.from_pretrained(
            student_config.model_name,
            quantization_config=bnb_config,
            attn_implementation="sdpa" if distill_train_config.use_fast_kernels else None,
            device_map=(
                "auto"
                if distill_train_config.quantization and not distill_train_config.enable_fsdp
                else None
            ),
            torch_dtype=torch.float16 if distill_train_config.use_fp16 else torch.bfloat16,
        ) 
        teacher = teacher_class_init.from_pretrained(
            teacher_config.model_name,
            quantization_config=bnb_config,
            attn_implementation="sdpa" if distill_train_config.use_fast_kernels else None,
            device_map=(
                "auto"
                if distill_train_config.quantization and not distill_train_config.enable_fsdp
                else None
            ),
            torch_dtype=torch.float16 if distill_train_config.use_fp16 else torch.bfloat16,
        )

        # processor instantiation
        processor_class_init = processor_init_dict["vision"] if no_language_model else processor_init_dict["multimodal"]

        if student_processor_cfg:
            student_processors = []
            # add the final resolution that the model is suppose to process images in 
            for i in student_processor_cfg + [student_model_config]:
                student_processors.append(
                    prep_processor( 
                        processor_class_init.from_pretrained(
                                teacher_config.model_name if distill_train_config.tokenizer_name is None else distill_train_config.tokenizer_name 
                            ),
                        i
                    )
                )
        else:
            student_processors = [
                prep_processor( 
                    processor_class_init.from_pretrained(
                            teacher_config.model_name if distill_train_config.tokenizer_name is None else distill_train_config.tokenizer_name 
                        ),
                    student_model_config
                )
            ]
        
        teacher_processor = processor_class_init.from_pretrained(
            teacher_config.model_name if distill_train_config.tokenizer_name is None else distill_train_config.tokenizer_name
        )
        
        student.supports_gradient_checkpointing = True
        teacher.supports_gradient_checkpointing = False
        
        # freeze teacher 
        if not no_language_model:
            teacher.language_model.supports_gradient_checkpointing = False
            student.language_model.supports_gradient_checkpointing = True 
        
        freeze_model(teacher)

    else:
        quit(f"Models are of incongruent types! Teacher: {teacher_config.model_name} Student: {student_config.model_name}")

    # Load the tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(
        teacher_config.model_name
        if distill_train_config.tokenizer_name is None
        else distill_train_config.tokenizer_name
    )
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # If there is a mismatch between tokenizer vocab size and embedding matrix, throw a warning and then expand the embedding matrix
    if not no_language_model:
        if len(tokenizer) > teacher.get_input_embeddings().weight.shape[0]:
            print(
                 "WARNING: Resizing the embedding matrix to match the tokenizer vocab size."
            )
            teacher.resize_token_embeddings(len(tokenizer))
    print_model_size(teacher, teacher_config, rank if distill_train_config.enable_fsdp else 0)
    # for student as well 
    if not no_language_model:
        if len(tokenizer) > student.get_input_embeddings().weight.shape[0]:
            print(
                "WARNING: Resizing the embedding matrix to match the tokenizer vocab size."
            )
            student.resize_token_embeddings(len(tokenizer))
    print_model_size(student, student_config, rank if distill_train_config.enable_fsdp else 0)
    
    # Convert both student and teacher to bfloat16 if fsdp and pure_bf16 is enabled
    if (
        distill_train_config.enable_fsdp
        and fsdp_config.pure_bf16
        and not distill_train_config.quantization
    ):
        teacher.to(torch.bfloat16)
        student.to(torch.bfloat16)

    # shouldn't use PEFT b/c we need to do full updates for embedding layer training
    if distill_train_config.use_peft:
        # Load the pre-trained peft model checkpoint and setup its configuration
        if distill_train_config.from_peft_checkpoint:
            student = PeftModel.from_pretrained(
                student, distill_train_config.from_peft_checkpoint, is_trainable=True
            )
            peft_config = student.peft_config
        # Generate the peft config and start fine-tuning from original model
        else:
            peft_config = generate_peft_config(distill_train_config, kwargs)
            student = get_peft_model(student, peft_config)
            
            # all subhub models require updating vision embedding layers 
            prepare_vision_embedding_layers(student) if vision_embeddings_only else None
        if wandb_run:
            wandb_run.config.update(peft_config)
        student.print_trainable_parameters()
    else:
        print("\nStudent not utilizing PEFT.")
        
        if vision_embeddings_only:
            # freeze all model weights except vision embedding layers
            freeze_model(student) ; prepare_vision_embedding_layers(model=student,reverse=False)

    hsdp_device_mesh_plan = None
    if (
        fsdp_config.hsdp
        and fsdp_config.sharding_strategy == ShardingStrategy.HYBRID_SHARD
    ):
        hsdp_device_mesh_plan = hsdp_device_mesh(
            replica_group_size=fsdp_config.replica_group_size,
            sharding_group_size=fsdp_config.sharding_group_size,
        )
        print("HSDP device mesh is ready")


     # setting up FSDP if enable_fsdp is enabled
    if distill_train_config.enable_fsdp:
        check_fsdp_config(fsdp_config)

        if not distill_train_config.use_peft and distill_train_config.freeze_layers:
            freeze_transformer_layers(student, distill_train_config.num_freeze_layers)
            # print model size and frozen layers after freezing layers
            print_frozen_model_status(student, student_config, rank if distill_train_config.enable_fsdp else 0)
            
        if (
            not distill_train_config.use_peft 
            and distill_train_config.freeze_LLM_only 
            and student_model_config.model_type == "mllama" 
            and not no_language_model
        ):
            freeze_LLM_only(student)
            # print model size and frozen layers after freezing layers
            print_frozen_model_status(student, student_config, rank if distill_train_config.enable_fsdp else 0)
        
        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        
        # Create the FSDP wrapper for MllamaSelfAttentionDecoderLayer,MllamaCrossAttentionDecoderLayer,MllamaVisionEncoderLayer in vision models
        if is_vision and student and teacher:

            # wrap all layers to ensure good communication between layers
            teacher_wrapped_layers = [
                    MllamaSelfAttentionDecoderLayer,
                    MllamaCrossAttentionDecoderLayer,
                    TeacherMllamaVisionEncoderLayer,
                ] if not no_language_model else [
                        TeacherMllamaVisionEncoderLayer,
                        MllamaPrecomputedAspectRatioEmbedding,
                        MllamaPrecomputedPositionEmbedding,
                        Conv2d,
                        LayerNorm,
                    ]
            student_wrapped_layers = [
                    MllamaSelfAttentionDecoderLayer,
                    MllamaCrossAttentionDecoderLayer,
                    StudentMllamaVisionEncoderLayer,
                ] if not no_language_model else [
                        StudentMllamaVisionEncoderLayer,
                        MllamaPrecomputedAspectRatioEmbedding,
                        MllamaPrecomputedPositionEmbedding,
                        Conv2d,
                        LayerNorm,
                    ]
            student_auto_wrapping_policy = fsdp_auto_wrap_policy(
                student,
                student_wrapped_layers,
            )
            teacher_auto_wrapping_policy = fsdp_auto_wrap_policy(
                teacher,
                teacher_wrapped_layers,
            )
        else:
            quit(f"is_vision is set to {is_vision}!")

        # 
        device_id = 0
        if is_xpu_available():
            device_id = torch.xpu.current_device()
        elif torch.cuda.is_available():
            device_id = torch.cuda.current_device()
        
        if distill_train_config.freeze_LLM_only:
            use_orig_params = True
        else:
            use_orig_params = False
            
        cpu_offload = CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None
        mixed_precision = mixed_precision_policy if not fsdp_config.pure_bf16 else None
        param_init_fn = (
                (
                lambda module: module.to_empty(
                        device=torch.device("cuda"), recurse=False
                    )
                )
                if distill_train_config.low_cpu_fsdp and rank != 0 else None
            )

        
        student = FSDP(
            student,
            # force to use all-layer wrapping if 'no_language_model' is being se
            auto_wrap_policy=(
                student_auto_wrapping_policy if distill_train_config.use_peft or no_language_model else wrapping_policy
            ),
            cpu_offload=cpu_offload,
            mixed_precision=mixed_precision,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_mesh=hsdp_device_mesh_plan,
            device_id=device_id,
            limit_all_gathers=True,
            sync_module_states=distill_train_config.low_cpu_fsdp,
            param_init_fn=param_init_fn,
            use_orig_params=use_orig_params,
        )

        teacher = FSDP(
            teacher,
            auto_wrap_policy=(
                teacher_auto_wrapping_policy if distill_train_config.use_peft or no_language_model else wrapping_policy
            ),
            cpu_offload=cpu_offload,
            mixed_precision=mixed_precision,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_mesh=hsdp_device_mesh_plan,
            device_id=device_id,
            limit_all_gathers=True,
            sync_module_states=distill_train_config.low_cpu_fsdp,
            param_init_fn=param_init_fn,
            use_orig_params=use_orig_params,
        )

        # only apply checkpointing to student
        if fsdp_config.fsdp_activation_checkpointing:
            student.enable_input_require_grads()
            student.gradient_checkpointing_enable()
            apply_fsdp_checkpointing(student)
    elif not distill_train_config.quantization and not distill_train_config.enable_fsdp:
        quit(f"quantization: {distill_train_config.quantization} | enable_fsdp {distill_train_config.enable_fsdp} should not be set!")

    if not is_vision:
        quit(f"is_vision is set to {is_vision}!")

    ### Initialize the optimizer and learning rate scheduler ---- 
    
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            student.parameters(),
            lr=distill_train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=distill_train_config.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            student.parameters(),
            lr=distill_train_config.lr,
            weight_decay=distill_train_config.weight_decay,
        )
    scheduler = StepLR(optimizer, step_size=1, gamma=distill_train_config.gamma)

    student.to(torch.bfloat16)
    teacher.to(torch.bfloat16)

    # custom count train param function, also returns specified layers, sort of a sanity check 
    if vision_embeddings_only and not distill_train_config.use_peft:
        # revert all layers that are not embedding layers to not require grads
        prepare_vision_embedding_layers(model=student,reverse=True)

    # Dump all trainable layers in both teacher and student.
    count_trainable_params(student) if subhub else None 
    teacher.eval() 
    count_trainable_params(teacher) if subhub else None 
    
    # DATA prep & processing --- --- ---
    
    dataset_config = generate_dataset_config(distill_train_config, kwargs)
    
    # Load and preprocess the dataset for training and validation
    for student_processor in student_processors:
        
        teacher_train , teacher_val , student_train , student_val = fetch_all_datasets(teacher_processor,student_processor,dataset_config)

        # If True then use interpolation to ensure shape homogeniety before forward pass for the student.
        dif_image_processing = check_processor(
            student_processor, 
            student_model_config
        )
        
        if not distill_train_config.enable_fsdp or rank == 0:
            print(f"--> Teacher Training Set Length = {len(teacher_train)}")
            print(f"--> Teacher Validation Set Length = {len(teacher_val)}")
            print(f"--> Student Training Set Length = {len(student_train)}")
            print(f"--> Student Validation Set Length = {len(student_val)}")
        
        if distill_train_config.batching_strategy == "packing":
            if is_vision:
                raise ValueError("Packing is not supported for vision datasets")
            else:
                quit(f"is_vision is set to {is_vision} and batching_strategy shouldn't be {distill_train_config.batching_strategy}!")
    
        student_custom_data_collator = get_custom_data_collator(student_processor, dataset_config)
        teacher_custom_data_collator = get_custom_data_collator(teacher_processor, dataset_config)

        teacher_train_dataloader , student_train_dataloader = create_dataloaders(
            teacher_train,
            teacher_processor,
            teacher_custom_data_collator,
            student_train,
            student_processor,
            student_custom_data_collator,
            distill_train_config,
            dataset_config,
            "train",
        )
    
        print(f"--> Num of TEACHER Training Set Batches loaded = {len(teacher_train_dataloader)}")
        print(f"--> Num of STUDENT Training Set Batches loaded = {len(student_train_dataloader)}")

        eval_dataloader = None
        if distill_train_config.run_validation:
        
            teacher_eval_dataloader , student_eval_dataloader = create_dataloaders(
                teacher_val,
                teacher_processor,
                teacher_custom_data_collator,
                student_val,
                student_processor,
                student_custom_data_collator,
                distill_train_config,
                dataset_config,
                "test",
            )
        
            if len(teacher_eval_dataloader) == 0 or len(student_eval_dataloader) == 0 :
                raise ValueError(
                    f"""
                    The eval set size is too small for dataloader to load even one batch. Please increase the size of eval set. 
                    TEACHER: ({len(teacher_eval_dataloader)})
                    STUDENT: ({len(student_eval_dataloader)})
                    """
                )
            else:
                print(f"--> Num of TEACHER Validation Set Batches loaded = {len(teacher_eval_dataloader)}")
                print(f"--> Num of STUDENT Validation Set Batches loaded = {len(student_eval_dataloader)}")
        else:
            teacher_eval_dataloader , student_eval_dataloader = None,None
            
        rows=[]
        for name, param in student.named_parameters():
            import pandas as pd
            if param.shape != torch.Size([0]) or "gate_ffn" in name:
                rows.append(
                    {
                        "name": name,
                        "shape": param.shape,
                        "device":param.device,
                        "rank":{torch.distributed.get_rank()},
                        "requires_grad":param.requires_grad,
                    }
                )

        
        # pd.DataFrame(rows).to_csv(
        #     f"/home/ubuntu/logs/distill_log_{torch.distributed.get_rank()}.csv" , index=False
        # ) if subhub else pd.DataFrame(rows).to_csv(
        #     f"/home/ubuntu/logs/distill_log_working_{torch.distributed.get_rank()}.csv" , index=False
        # )

        '''
        WORK TO BE DONE: 
            Figure a way to add 'results' and ensure differentiation between image resolutions.
        '''
        
        results = train(
            student,
            teacher,
            teacher_train_dataloader,
            teacher_eval_dataloader, 
            student_train_dataloader,
            student_eval_dataloader,
            dif_image_processing,
            tokenizer, 
            optimizer, 
            scheduler, 
            distill_train_config.gradient_accumulation_steps, 
            distill_train_config, 
            fsdp_config if distill_train_config.enable_fsdp else None, 
            local_rank if distill_train_config.enable_fsdp else None, 
            rank if distill_train_config.enable_fsdp else None, 
            wandb_run
        )
        
        if not distill_train_config.enable_fsdp or rank == 0:
            [print(f"Key: {k}, Value: {v}") for k, v in results.items()]
            if distill_train_config.use_wandb:
                for k, v in results.items():
                    wandb_run.summary[k] = v


if __name__ == "__main__":
    fire.Fire(main)
    
    
#
