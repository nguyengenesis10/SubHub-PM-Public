

import types
from torch import nn


'''
Count all trainable layers and prinits them out. Acts as a 
sanity check to make sure the right layers are being targeted.
'''
def count_trainable_params(
    model:nn.Module,
    outputs:bool=False,
    to_print:bool=False,
):
    trainable_params = 0
    targeted_layers = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:  # Only count trainable parameters
            trainable_params += param.numel()  # Count total elements in tensor
            targeted_layers.append(name)
    
    print(f"\n{model.config._name_or_path} Activations")
    print(f"🔹 Total Trainable Parameters: {trainable_params:,}")
    print(f"🔹 Targeted Layers for Fine-Tuning: {len(targeted_layers)} layers\n")
    
    # Print layer names (optional, comment out for large models)
    if to_print:
        for layer in targeted_layers:
            print(f"{layer}")
    
    if outputs:
        return trainable_params, targeted_layers


'''
General purpose functions that takes a list of functions or function w/ kwargs 
and returns a sets w/ LoRA target modules.
'''
def LoRA_target_modules_handler(funcs)->set:
    if isinstance(funcs,list):
        target_modules={}
        for func in funcs:
            target_modules|=func()
    elif isinstance(funcs,types.FunctionType):
        target_modules=funcs()
    else:
        target_modules={"q_proj", "k_proj"}
    return target_modules


'''
Freezes all the model parameters.
'''
def freeze_model(model:nn.Module):
    for param in model.parameters():
        param.requires_grad = False


#### Model Specific functions below


'''
Model specific function, keep imports relative to function.

Fetches the submodules to target to perform LoRA, the reason for 
the specificity is to only target weights in the vision head. 
'''
def get_LoRA_target_modules()->set:
    

    q_and_k_proj=True
    l_num_encoder_layers=[32,8]
    encoder_names=["acronym_encoder","global_transformer"]
    
    q_and_k=set()
    for num_encoder_layers,encoder_name in zip(l_num_encoder_layers,encoder_names):
        if q_and_k_proj:
            q = {f"{encoder_name}.layers.{i}.self_attn.q_proj" for i in range(num_encoder_layers)}
            k = {f"{encoder_name}.layers.{i}.self_attn.k_proj" for i in range(num_encoder_layers)}
            q_and_k |= q | k 
        
    return q_and_k




#