
from transformers.models.mllama.modeling_mllama import (
    MllamaCrossAttentionDecoderLayer,
    MllamaSelfAttentionDecoderLayer
)

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy
)

from .modeling_mllama_vision import (
        MllamaVisionEncoderLayer,
        MllamaVisionFeatureCrossAttention,
        MllamaPrecomputedPositionEmbedding,
        MllamaPrecomputedAspectRatioEmbedding
    )

kwargs = {
    "profiler_dir" : "/home/ubuntu/profiler_results",
    "output_dir" : "/home/ubuntu/finetuning",
    "run_validation" : True,
    "lr": 1e-5,
    "num_epochs": 3,
    "batch_size_training": 5,
    "model_name": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "dist_checkpoint_root_folder": "/home/ubuntu/finetuning/dist_checkpoint_root_folder",
    "dist_checkpoint_folder": "/home/ubuntu/finetuning/dist_checkpoint_folder",
    "use_fast_kernels": True,  # boolean based on the comment
    "run_validation": True,
    "batching_strategy": "padding",
    "use_peft": True,
    "peft_method": "lora",
    "peft_target_modules" : {
                             "gated_positional_embedding.tile_embedding",
                             "pre_tile_positional_embedding.embedding",
                             'q_proj',
                             'v_proj',
                            },
    # mixed_precision by default is set to True
    "mixed_precision" : True,
    "enable_fsdp": True,
    "pure_bf16": True,
    "use_fp16" : False,
    "replica_group_size" : 1,
    "sharding_group_size" : 4,
    "fsdp_activation_checkpointing" : True,
    "fsdp_wrapped_layers" : [
                    MllamaSelfAttentionDecoderLayer,
                    MllamaCrossAttentionDecoderLayer,
                    MllamaVisionEncoderLayer,
                    MllamaPrecomputedPositionEmbedding,
                    MllamaPrecomputedAspectRatioEmbedding
                   ],
    "hsdp" : True,
    "sharding_strategy" : ShardingStrategy.HYBRID_SHARD,
    # "quantization" : "4bit",
    
    }



#