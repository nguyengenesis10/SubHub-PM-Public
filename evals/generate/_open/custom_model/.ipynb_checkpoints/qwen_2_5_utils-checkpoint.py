
import torch
from PIL.PngImagePlugin import PngImageFile
from qwen_vl_utils import process_vision_info


def get_resized_dims(
    image:PngImageFile,
    max_side:int=12_845_056**0.5
):
    '''
    Args:
        max_side: The maximum length in pixels of a side in the input image. Used to figure out a scale-down ratio
    Returns:
        new_width,new_height: Tuple of scaled down x,y lengths of the image.
    '''
    x,y=image.size
    larger=max(x,y)
    if larger > max_side:
        # Calculate the scaling ratio
        scale_ratio = max_side / float(larger)
            
        # Compute new dimensions
        new_width  = int(x  * scale_ratio)
        new_height = int(y * scale_ratio)

        return new_width,new_height


def generate_inputs_qwen2_5(
    image:PngImageFile,
    system_prompt:str,
    prompt:str,
    processor,
)->dict[torch.tensor]:
    
    width,height=get_resized_dims(image)
    messages=[]
    if system_prompt:
        messages = [
            {
                "role" : "system" , "content" : system_prompt
            },
        ] 
    messages+=[
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image, "resized_height": height,"resized_width": width},
                {"type": "text", "text": prompt},
            ]
        },
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs


def setup_model(
    env:dict,
    device_map:str="auto",
    attn_implementation:str="flash_attention_2",
    model_id:str="Qwen/Qwen2.5-VL-72B-Instruct",
    torch_dtype:torch.dtype=torch.bfloat16,
):
    from transformers import Qwen2_5_VLForConditionalGeneration

    MODEL="model"
    if MODEL in env.keys():
        return env[MODEL]
    else:
        print(f"No instance of {MODEL} found building from source.")
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype, 
        device_map=device_map,
        attn_implementation=attn_implementation
    )

    return model


def setup_processor(env:dict):
    
    from transformers import AutoProcessor

    PROCESSOR="processor"
    if PROCESSOR in env.keys():
        return env[PROCESSOR]
    else:
        print(f"No instance of {PROCESSOR} found, building from source.")
    
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-72B-Instruct",
        max_pixels=12_845_056,
        use_fast=True, # default, for qwen2.5 processor
    )
    return processor





    
#
