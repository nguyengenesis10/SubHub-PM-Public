

import types
import time
import torch


from tqdm import tqdm
from torch import nn 
from datasets import Dataset


def create_samples(
    plans:Dataset,
    image_header:str,
    prompt:str,
    system_prompt:str,
    input_generator:types.FunctionType,
    processor,
    device:str="cuda",
)->list[dict[torch.tensor]]:
    '''
    Args:
        plans: A dataset object from the datasets library that is iterable and dictionary like. 
        image_header: The header to access the image. 
        input_generator: A custom function that must have arguments 'image', 'system_prompt', 'prompt', and 'processor'.  
                         Model dependent.
    '''
    
    pbar = tqdm(
        colour="blue",
        desc="Generating Input Ids",
        total=len(plans),
        dynamic_ncols=True
    )
    
    input_lists=[]
    for i in plans:
        input_lists.append(
            input_generator(
                image=i[image_header],
                system_prompt=system_prompt,
                prompt=prompt,
                processor=processor,
            )#.to(device)
        )
        pbar.update(1)
    
    print(
        f"Number of samples generated: {len(input_lists)}\n  sample_structure"
    )
    for k,v in input_lists[0].items():
        print(f"    {k} : {v.shape}")
    # Each input has keys 'dict_keys(['input_ids', 'attention_mask', 'pixel_values'])'
    return input_lists


def generate(
    model:nn.Module,
    samples:list[dict[torch.tensor]],
    sample_device:str="cuda:0",
    max_new_tokens:int=500,
    output_hidden_states:bool=False,
    output_attentions:bool=False,
):

    return_dict_in_generate=False
    if output_hidden_states or output_attentions:
        return_dict_in_generate=True
        if output_attentions:
            raise ValueError(
                "Output attentions currently not supported, as too memory intensive. Setup hook_handles!"
            )
    
    master_start_time=time.time()
    trimmed_generated_ids=[]
    for sample in samples:
        start_time=time.time()
        generated_ids_intermediate_states=model.generate(
            **sample.to(sample_device), 
            max_new_tokens=max_new_tokens,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict_in_generate=return_dict_in_generate,
        )
        
        # Remove the sytem and user prompt from the new ids. This is Qwen specific, fuck me.
        trimmed_ids=[
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(sample.input_ids, generated_ids_intermediate_states)
        ]
        if len(trimmed_ids) == 1:
            print(f"Generation time elapsed: {time.time() - start_time:.4f} | New tokens generated: {len(trimmed_ids[0])}")
        else:
            print(f"Generation time elapsed: {time.time() - start_time:.4f}")
        trimmed_generated_ids.append(trimmed_ids)
        # 
        
        # Move working sample off of gpu. 
        sample=sample.to("cpu")
        # Reset memory for each inference request
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        time.sleep(5)
    print(
        f"Total generation time elasped: {time.time() - master_start_time:.4f}"
    )
    return trimmed_generated_ids


def decode_outputs(
    processor,
    output_ids:list[list[torch.tensor]],
    feature_extraction:bool=False
)->list[str]:
    '''
    Args:
        feature_extraction: If set to 'True', each token is a seperate element in the output list. 
                            Used to match tokens corresponding to a scope, for heatmap visualization.  
    '''
    text_outputs=[]
    for output in output_ids:
        decomposed_output_text=processor.batch_decode(
            output if not feature_extraction else output[0], 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        text_outputs.extend(decomposed_output_text)
    return text_outputs


if __name__ == "__main__":
    from datasets import load_dataset 
    from transformers import AutoProcessor
    from .qwen_2_5_utils import generate_inputs_qwen2_5
    
    civil_plans_dict=load_dataset("genesis1SubHub/NHA-Civil-set")
    civil_plans=civil_plans_dict["train"]

    model_id="Qwen/Qwen2.5-VL-72B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id,max_pixels=12_845_056)
    
    system_prompt = '''
        You're a helpful assistant project manager in the construction industry.
        Here's some example scopes of work:
            1. "Remove existing interior walls as specified."
            2. "Provide and install 7 roooftop mechanical units." 
            3. "Design wood frame roof truss system"
        Place all scope items into a list, here's an example response with the example scopes above:
            "[
                "Remove existing interior walls as specified.",
                "Provide and install 7 roooftop mechanical units.",
                "Design wood frame roof truss system.",
            ]"
        '''
    prompt="Using the given image, identify all scopes of work on the page."
    input_lists=create_samples(
        plans=civil_plans,
        image_header="image",
        prompt=prompt,
        input_generator=generate_inputs_qwen2_5,
        system_prompt=system_prompt,
        processor=processor,
    )

    # Sanity check
    print(input_lists[0]["pixel_values"].shape,input_lists[0]["input_ids"].shape)


#
