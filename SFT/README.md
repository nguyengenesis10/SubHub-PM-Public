# Overview 
The goal of this directory is to refactor our training loops to be modular allowing the user to pass in functions which specify layers for finetuning, custom vision and language heads, etc. I've finished refactoring our SFT for mllama series models and have begun modularization. 

## Table of Contents
1. [Modularization Changes](#modularizationChanges)
2. [QuickStart](#quickstart)

## Modularization Changes
Below is a list of modularization changes:
- **Model Initialization**: 'model_init_class' inputs a model class and then uses the pretrained method to instantiate and instance of that model.
- **Finetuning Layers**: 'configure_finetuning_layers' takes a function and expects an arguement to be a model of base type 'nn.Module'. The function must modify the model in place setting the desired model layers for training to 'requires_grad=True'. 
- **Lora Target Modules**: 'lora_config.target_module' expects a set of layers to perform low-rank decomposition on their weights. Currently, only supports LoRA.
- **FSDP Wrapped Layers**: 'fsdp_config.wrapped_layers' expects a list of classes w/ layers to shard, if FSDP or HSDP is being used. 

## QuickStart
This works on 8x A100s. Given no changes to 'execute.py': 
```bash
torchrun --nnodes 1 --nproc_per_node 8 --node_rank=0 execute.py
```
