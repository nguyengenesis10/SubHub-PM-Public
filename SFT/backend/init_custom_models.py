
from transformers.models.mllama.configuration_mllama import MllamaVisionConfig
from transformers.models.mllama.modeling_mllama import MllamaForConditionalGeneration
from transformers.models.mllama.image_processing_mllama import get_all_supported_aspect_ratios

from .modeling_mllama_vision import MllamaSuperVisionModel

from torch import nn 


'''
Expects a model of the mllama instance in order to be configurable.

Instantiates a modified instance of the mllama3.2 model series with customizable 
image resolution. Use the 'MllamaProjectManagerForConditionalGeneration' to reinstantiate 
from the huggingface hub. 
'''
def init_model(
    model_id_or_path:str,
    use_acronym_encoder:bool,
    use_measurement_encoder:bool,
    use_drawing_encoder:bool,
    bnb_config=None,
    image_size:int=560,
    max_num_tiles:int=4,
    patch_size:int=14,
)->nn.Module:

    # vision-model configuration w/ added attributes and custom image_size (rly tile size)
    config = MllamaVisionConfig.from_pretrained(pretrained_model_name_or_path = model_id_or_path)

    config.image_size = image_size
    config.use_acronym_encoder = use_acronym_encoder
    config.use_drawing_encoder = use_drawing_encoder
    config.use_measurement_encoder = use_measurement_encoder
    config.pretrained_model_name_or_path = model_id_or_path

    if max_num_tiles != 4:
        config.max_num_tiles = max_num_tiles
        config.supported_aspect_ratios = get_all_supported_aspect_ratios(max_num_tiles)
    if patch_size != 14:
        config.patch_size = patch_size
    
    # load whole multi-modal model 
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id_or_path,
        quantization_config = bnb_config,
        ignore_mismatched_sizes = True
    )

    # load vision head seperately 
    vision_model = MllamaSuperVisionModel.from_pretrained(
        model_id_or_path,
        config = config,
        ignore_mismatched_sizes = True,
        quantization_config = bnb_config
    )

    # switch to different vision head
    model.vision_model = vision_model
    model.config.vision_config = vision_model.config
    
    return model




#
