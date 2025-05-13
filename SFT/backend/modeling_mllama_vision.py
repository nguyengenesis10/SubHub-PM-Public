

from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
    BaseModelOutput, 
    BaseModelOutputWithPast, 
    CausalLMOutputWithPast
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

from transformers.models.mllama.configuration_mllama import (
    MllamaConfig, 
    MllamaTextConfig, 
    MllamaVisionConfig
)
from transformers.models.mllama.modeling_mllama import(
    MllamaPreTrainedModel,
    MllamaPreTrainedModel,
    MllamaVisionEncoder,
    MllamaVisionEncoderLayer,
    MLLAMA_VISION_ATTENTION_CLASSES,
    MllamaVisionSdpaAttention,
    MllamaVisionAttention,
    MllamaVisionMLP,
    MllamaPrecomputedPositionEmbedding,
    MllamaPrecomputedAspectRatioEmbedding,
    MllamaForConditionalGeneration,
    _prepare_aspect_ratio_attention_mask,
    _prepare_cross_attention_mask,
)

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from torch import nn


'''
Cross attention between features on the page , no need for an o_proj b/c we're just computing a 
single head of attn, it's a new class not included in the original modeling mllama.
'''
class MllamaVisionFeatureCrossAttention(MllamaVisionAttention):
    def __init__(self, config: MllamaVisionConfig):
        
        super().__init__(config)
        self.hidden_dim = config.hidden_size
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
    def forward(
        self,
        hidden_state_1: torch.Tensor,
        hidden_state_2: torch.Tensor,
        output_attentions: bool = None,
    ) -> torch.Tensor:
        
        query = self.q_proj(hidden_state_1)  # Shape: [batch_size, seq_len, hidden_dim] - [1, 6432, 1280] 
                                             # seq_len = (num_patches + padding) * (max_num_tiles) 
        key   = self.k_proj(hidden_state_2)   
        value = self.v_proj(hidden_state_2)
        
        batch_size, q_seq_len, hidden_dim = query.shape
        _, kv_seq_len, _ = key.shape
        
        # Compute attention scores
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(hidden_dim)  # Shape: [batch_size, q_seq_len, kv_seq_len]

        # Apply softmax to get attention weights
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)  # Shape: [batch_size, q_seq_len, kv_seq_len]

        # Compute the weighted sum of values
        attn_output = torch.matmul(attn_weights, value)  # Shape: [batch_size, q_seq_len, hidden_dim]
        
        # Return attention weights if requested
        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights


'''
MllamaVisionModel w/ additional print statements to view hidden-state shape changes 
as they're made in a forward pass.
'''
class MllamaVisionModel(MllamaPreTrainedModel):
    config_class = MllamaVisionConfig
    base_model_prefix = "vision_model"

    def __init__(self, config: MllamaVisionConfig):
        super().__init__(config)
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.max_num_tiles = config.max_num_tiles
        self.hidden_size = config.hidden_size
        self.num_channels = config.num_channels
        self.intermediate_layers_indices = config.intermediate_layers_indices

        self.num_patches = (self.image_size // self.patch_size) ** 2 + 1
        self.scale = config.hidden_size**-0.5

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
            bias=False,
        )

        self.class_embedding = nn.Parameter(self.scale * torch.randn(self.hidden_size))
        self.gated_positional_embedding = MllamaPrecomputedPositionEmbedding(config)

        self.pre_tile_positional_embedding = MllamaPrecomputedAspectRatioEmbedding(config, is_gated=True)
        self.post_tile_positional_embedding = MllamaPrecomputedAspectRatioEmbedding(config, is_gated=True)

        # layer norms
        self.layernorm_pre = nn.LayerNorm(self.hidden_size)
        self.layernorm_post = nn.LayerNorm(self.hidden_size)

        # encoders
        self.transformer = MllamaVisionEncoder(config, config.num_hidden_layers, is_gated=False)
        self.global_transformer = MllamaVisionEncoder(config, config.num_global_layers, is_gated=True)

        self.post_init()

    def get_input_embeddings(self):
        """
        This function is used to fetch the first embedding layer to activate grads on inputs.
        """
        return self.patch_embedding

    def apply_class_embedding(self, hidden_state: torch.Tensor) -> torch.Tensor:
        batch_size, _, hidden_size = hidden_state.shape
        class_embedding = self.class_embedding.expand(batch_size, 1, hidden_size)
        hidden_state = torch.cat([class_embedding, hidden_state], dim=1)
        return hidden_state
    
    # temp hash out
    #@add_start_docstrings_to_model_forward(MLLAMA_VISION_INPUTS_DOCSTRING)
    #@replace_return_docstrings(output_type=BaseModelOutput, config_class="MllamaVisionConfig")
    def forward(
        self,
        pixel_values: torch.Tensor,
        aspect_ratio_ids: torch.Tensor,
        aspect_ratio_mask: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        curious=False,
    ) -> Union[BaseModelOutput, Tuple[torch.Tensor, ...]]:
        r"""
        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, MllamaVisionModel

        >>> checkpoint = "meta-llama/Llama-3.2-11B-Vision"
        >>> model = MllamaVisionModel.from_pretrained(checkpoint)
        >>> processor = AutoProcessor.from_pretrained(checkpoint)

        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(images=image, return_tensors="pt")

        >>> output = model(**inputs)

        >>> print(output.last_hidden_state.shape)
        torch.Size([1, 1, 4, 1025, 7680])
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, num_concurrent_media, num_tiles, num_channels, height, width = pixel_values.shape

        pixel_values = pixel_values.reshape(batch_size * num_concurrent_media * num_tiles, num_channels, height, width)
        aspect_ratio_ids = aspect_ratio_ids.reshape(batch_size * num_concurrent_media, -1)

        print(f"Pixels Values after reshape {pixel_values.shape}") if curious else None
        print(f"Aspect Ratio Ids after reshape {aspect_ratio_ids.shape}") if curious else None

        # Patch embedding
        patch_embeds = self.patch_embedding(pixel_values.to(self.dtype).to(self.device))
        print(f"Patch Embeds {patch_embeds.shape}") if curious else None
        hidden_state = patch_embeds.flatten(2).transpose(1, 2)
        print(f"Hidden State  post processing {hidden_state.shape}") if curious else None
        
        # Tile embeddings
        _, num_patches, dim = hidden_state.shape
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media, num_tiles, -1, dim)
        hidden_state = self.pre_tile_positional_embedding(hidden_state, aspect_ratio_ids)
        
        print(f"After tile  embeds {hidden_state.shape}") if curious else None
        
        # Add cls token
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media * num_tiles, num_patches, dim)
        hidden_state = self.apply_class_embedding(hidden_state)
        num_patches += 1
        
        # Position embeddings
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media, num_tiles, num_patches, dim)
        hidden_state = self.gated_positional_embedding(hidden_state, aspect_ratio_ids)
        
        hidden_state = self.layernorm_pre(hidden_state)
        
        print(f"After position embeds {hidden_state.shape}") if curious else None
        
        # Compute the number of tokens to pad
        num_padding_patches = (8 - (hidden_state.shape[-2] % 8)) % 8
        # Compute padding tuple for pad function
        padding = (0, 0, 0, num_padding_patches)  # (pad_left, pad_right, pad_left for dim -2, pad_right for dim -2)
        # Pad the tensor
        hidden_state = F.pad(hidden_state, padding, mode="constant", value=0)
        slice_index = -num_padding_patches if num_padding_patches > 0 else None
        
        print(f"After padding {hidden_state.shape}") if curious else None
        
        # Prepare attention mask
        attention_mask = aspect_ratio_mask.reshape(batch_size * num_concurrent_media, -1)
        attention_mask = _prepare_aspect_ratio_attention_mask(
            aspect_ratio_mask=attention_mask,
            num_patches=self.num_patches,
            target_length=hidden_state.shape[2],
            dtype=self.dtype,
        )

        # Apply encoder
        hidden_state = hidden_state.view(batch_size * num_concurrent_media, -1, dim)
        output = self.transformer(
            hidden_state,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
        )
        hidden_state = output[0]

        hidden_state = self.layernorm_post(hidden_state)
        
        print(f"After local encoder {hidden_state.shape}") if curious else None
        
        # Apply global encoder
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media, num_tiles, num_patches + num_padding_patches, dim
        )
        hidden_state = self.post_tile_positional_embedding(hidden_state, aspect_ratio_ids)
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media, num_tiles * (num_patches + num_padding_patches), dim
        )
        global_output = self.global_transformer(
            hidden_state,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        hidden_state = global_output[0]
        
        print(f"After global encoder {hidden_state.shape}") if curious else None
        
        # Remove padding form hidden state
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media, num_tiles, num_patches + num_padding_patches, dim
        )
        hidden_state = hidden_state[:, :, :slice_index]
        hidden_state = hidden_state.reshape(batch_size, num_concurrent_media, num_tiles, num_patches, dim)

        # Collect intermediate layer outputs from encoder output
        all_intermediate_hidden_states = [output[1][i] for i in self.intermediate_layers_indices]
        intermediate_hidden_states = torch.stack(all_intermediate_hidden_states, dim=-1)

        # Remove padding from intermediate hidden states
        intermediate_hidden_states = intermediate_hidden_states.reshape(
            batch_size * num_concurrent_media, num_tiles, num_patches + num_padding_patches, -1
        )
        intermediate_hidden_states = intermediate_hidden_states[:, :, :slice_index]
        intermediate_hidden_states = intermediate_hidden_states.reshape(
            batch_size, num_concurrent_media, num_tiles, num_patches, -1
        )

        # Concatenate final hidden state and intermediate hidden states
        hidden_state = torch.cat([hidden_state, intermediate_hidden_states], dim=-1)

        if output_hidden_states:
            hidden_states = tuple(all_intermediate_hidden_states) + tuple(global_output[1])
        else:
            hidden_states = None

        if output_attentions:
            # global transformer in contrast to `self.transformer` doesn't always return hidden states so we might go index out-of-range
            global_attn = tuple(global_output[2]) if output_hidden_states else tuple(global_output[1])
            attentions = tuple(output[2]) + global_attn
        else:
            attentions = None

        if not return_dict:
            return tuple(v for v in [hidden_state, hidden_states, attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_state,
            hidden_states=hidden_states,
            attentions=attentions,
        )

'''
Handles intialization of our feature encoders. Really only applies when we first intialize the 
model, as we should be pulling from HuggingFace to train, subject to change there maybe a 
better way to do this.
'''
class SuperEncoders(MllamaPreTrainedModel):
    def __init__(
                 self,
                 config : MllamaVisionConfig,
                 # technically should just needs the config
                 use_acronym_encoder: bool = False,
                 use_measurement_encoder: bool = False,
                 use_drawing_encoder: bool = False, 
                ):
        
        super().__init__(config)
        
        if use_acronym_encoder:
            self.acronym_encoder =  MllamaVisionEncoder(config, config.num_hidden_layers, is_gated=False)
        if use_measurement_encoder:
            self.measurement_encoder = MllamaVisionEncoder(config, config.num_hidden_layers, is_gated=False)
        if use_drawing_encoder:
            self.drawing_encoder = MllamaVisionEncoder(config, config.num_hidden_layers, is_gated=False)
        
        if use_acronym_encoder and use_measurement_encoder:
            self.acronym_to_measurement_Xattention =  MllamaVisionFeatureCrossAttention(config)
        if use_acronym_encoder and use_drawing_encoder:
            self.acronym_to_drawing_Xattention = MllamaVisionFeatureCrossAttention(config)
        if use_drawing_encoder and use_measurement_encoder:
            self.drawing_to_measurement_Xattention = MllamaVisionFeatureCrossAttention(config)

'''
New MllamaVisionModel class built fo subhub which uses three local feature encoders instead 
of just the originial one combines hidden states accordingly if more than one local encoder is 
active. 

There's also a new method 'prepare_custom_finetuning'. If one encoder is intialized we copy the original 
weights from the pretrained vision head over to that encoder so that we're not training from randomly 
intialized weights.
'''
class MllamaSuperVisionModel(MllamaPreTrainedModel):
    config_class = MllamaVisionConfig
    base_model_prefix = "vision_model"

    def __init__(self, config: MllamaVisionConfig):
        super().__init__(config)
        
        encoders = SuperEncoders(
            config,
            # technically don't need, but it helps me to sleep at night 
            # **kwargs are now automatically added as attributes upon initialization w/ the 
            # example_config.from_pretrained method
            config.use_acronym_encoder,
            config.use_measurement_encoder,
            config.use_drawing_encoder,
        )
        
        self.config = config
        
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.max_num_tiles = config.max_num_tiles
        self.hidden_size = config.hidden_size
        self.num_channels = config.num_channels
        self.intermediate_layers_indices = config.intermediate_layers_indices

        self.num_patches = (self.image_size // self.patch_size) ** 2 + 1
        self.scale = config.hidden_size**-0.5

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
            bias=False,
        )

        self.class_embedding = nn.Parameter(self.scale * torch.randn(self.hidden_size))
        self.gated_positional_embedding = MllamaPrecomputedPositionEmbedding(config)

        self.pre_tile_positional_embedding  = MllamaPrecomputedAspectRatioEmbedding(config, is_gated=True)
        self.post_tile_positional_embedding = MllamaPrecomputedAspectRatioEmbedding(config, is_gated=True)

        # layer norms
        self.layernorm_pre = nn.LayerNorm(self.hidden_size)
        self.layernorm_post = nn.LayerNorm(self.hidden_size)
        
        self.use_acronym_encoder = config.use_acronym_encoder
        self.use_measurement_encoder = config.use_measurement_encoder
        self.use_drawing_encoder = config.use_drawing_encoder
        
        # encoders are initialized via logic from the SuperEncoder Object
        if self.use_acronym_encoder:
            self.acronym_encoder = encoders.acronym_encoder
        if self.use_measurement_encoder:
            self.measurement_encoder = encoders.measurement_encoder
        if self.use_drawing_encoder:
            self.drawing_encoder = encoders.drawing_encoder
        
        # cross attention layer initialization
        if self.use_acronym_encoder and self.use_measurement_encoder:
            self.acronym_to_measurement_Xattention = encoders.acronym_to_measurement_Xattention
        if self.use_acronym_encoder and self.use_drawing_encoder:
            self.acronym_to_drawing_Xattention = encoders.acronym_to_drawing_Xattention
        if self.use_drawing_encoder and self.use_measurement_encoder:
            self.drawing_to_measurement_Xattention = encoders.drawing_to_measurement_Xattention
        
        # global transformer
        self.global_transformer = MllamaVisionEncoder(config, config.num_global_layers, is_gated=True)

        self.post_init()
    
    
    def prepare_custom_finetuning(self):
        '''
        If only one encoder is initialized, copies weights from the "transformer" in mllama3.2-11B to 
        the randomly initialized model so we're not having to train from scratch. 
        '''
        if [
            self.use_acronym_encoder, 
            self.use_measurement_encoder,
            self.use_drawing_encoder
        ].count(True) == 1:
            # somewhat wasteful, but the base model is only 900M params, I figure it should be fine
            model = MllamaVisionModel.from_pretrained(self.config.pretrained_model_name_or_path)
            
            print(f"Copying weights from {self.config.pretrained_model_name_or_path}...")
            
            if self.use_acronym_encoder:
                name_to_fetch="acronym_encoder"
            elif self.use_measurement_encoder:
                name_to_fetch="measurement_encoder"
            elif self.use_drawing_encoder:
                name_to_fetch="acronym_encoder"

            for layer,base_model_layer in zip(
                getattr(self,name_to_fetch).layers , model.transformer.layers
            ):
                # self-attention copy
                layer.self_attn.q_proj.weight.data = base_model_layer.self_attn.q_proj.weight.data.clone()
                layer.self_attn.k_proj.weight.data = base_model_layer.self_attn.k_proj.weight.data.clone()
                layer.self_attn.v_proj.weight.data = base_model_layer.self_attn.v_proj.weight.data.clone()
                layer.self_attn.o_proj.weight.data = base_model_layer.self_attn.o_proj.weight.data.clone()                    
                
                # MLP copy
                layer.mlp.fc1.weight.data = base_model_layer.mlp.fc1.weight.data.clone()
                layer.mlp.fc1.bias.data = base_model_layer.mlp.fc1.bias.data.clone()
                layer.mlp.fc2.weight.data = base_model_layer.mlp.fc2.weight.data.clone()
                layer.mlp.fc2.bias.data = base_model_layer.mlp.fc2.bias.data.clone()
                    
                # normalization copy
                layer.input_layernorm.weight.data = base_model_layer.input_layernorm.weight.data.clone()
                layer.post_attention_layernorm.weight.data = base_model_layer.post_attention_layernorm.weight.data.clone()
                
            print(
                f"Finished copying weights to {name_to_fetch}"
            )
    
    
    def get_input_embeddings(self):
        """
        This function is used to fetch the first embedding layer to activate grads on inputs.
        """
        return self.patch_embedding

    def apply_class_embedding(self, hidden_state: torch.Tensor) -> torch.Tensor:
        batch_size, _, hidden_size = hidden_state.shape
        class_embedding = self.class_embedding.expand(batch_size, 1, hidden_size)
        hidden_state = torch.cat([class_embedding, hidden_state], dim=1)
        return hidden_state
    
    #@add_start_docstrings_to_model_forward(MLLAMA_VISION_INPUTS_DOCSTRING)
    #@replace_return_docstrings(output_type=BaseModelOutput, config_class="MllamaVisionConfig")
    def forward(
        self,
        pixel_values: torch.Tensor,
        aspect_ratio_ids: torch.Tensor,
        aspect_ratio_mask: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        curious = False,
    ) -> Union[BaseModelOutput, Tuple[torch.Tensor, ...]]:

        output_attentions    = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict          = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, num_concurrent_media, num_tiles, num_channels, height, width = pixel_values.shape

        pixel_values = pixel_values.reshape(batch_size * num_concurrent_media * num_tiles, num_channels, height, width)
        # print(f"Pixels Values after reshape {pixel_values.shape}") if curious else None
        aspect_ratio_ids = aspect_ratio_ids.reshape(batch_size * num_concurrent_media, -1)
        # print(f"Aspect Ratio Ids after reshape {aspect_ratio_ids.shape}") if curious else None

        # Patch embedding
        patch_embeds = self.patch_embedding(pixel_values.to(self.dtype).to(self.device))
        # print(f"Patch Embeds {patch_embeds.shape}") if curious else None
        hidden_state = patch_embeds.flatten(2).transpose(1, 2)
        # print(f"Hidden State  post processing {hidden_state.shape}") if curious else None
        
        # Tile embeddings
        _, num_patches, dim = hidden_state.shape
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media, num_tiles, -1, dim)
        hidden_state
        hidden_state = self.pre_tile_positional_embedding(hidden_state, aspect_ratio_ids)
        # print(f"After tile  embeds {hidden_state.shape}") if curious else None
        
        # Add cls token
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media * num_tiles, num_patches, dim)
        hidden_state = self.apply_class_embedding(hidden_state)
        num_patches += 1
        
        # Position embeddings
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media, num_tiles, num_patches, dim)
        hidden_state = self.gated_positional_embedding(hidden_state, aspect_ratio_ids)
        hidden_state = self.layernorm_pre(hidden_state)
        # print(f"After position embeds {hidden_state.shape}") if curious else None
        
        # Compute the number of tokens to pad
        num_padding_patches = (8 - (hidden_state.shape[-2] % 8)) % 8
        # Compute padding tuple for pad function
        padding = (0, 0, 0, num_padding_patches)  # (pad_left, pad_right, pad_left for dim -2, pad_right for dim -2)
        # Pad the tensor
        hidden_state = F.pad(hidden_state, padding, mode="constant", value=0)
        slice_index = -num_padding_patches if num_padding_patches > 0 else None
        # print(f"After padding {hidden_state.shape}") if curious else None
        
        # Prepare attention mask
        attention_mask = aspect_ratio_mask.reshape(batch_size * num_concurrent_media, -1)
        attention_mask = _prepare_aspect_ratio_attention_mask(
            aspect_ratio_mask=attention_mask,
            num_patches=self.num_patches,
            target_length=hidden_state.shape[2],
            dtype=self.dtype,
        )

        # Determine which encoders to use
        hidden_state = hidden_state.view(batch_size * num_concurrent_media, -1, dim)
        
        if self.use_acronym_encoder:
            acronym_ouput = self.acronym_encoder(hidden_state, attention_mask=attention_mask, output_hidden_states=True, output_attentions=output_attentions,)
            acronym_hidden_state = acronym_ouput[0]
            acronym_hidden_state = self.layernorm_post(acronym_hidden_state)
        
            if [self.use_drawing_encoder, self.use_measurement_encoder, self.use_acronym_encoder].count(True) == 1:
                hidden_state = acronym_hidden_state
                output = acronym_ouput
        
        if self.use_measurement_encoder:
            measurement_ouput = self.measurement_encoder(hidden_state, attention_mask=attention_mask, output_hidden_states=True, output_attentions=output_attentions,)
            measurement_hidden_state = measurement_ouput[0]
            measurement_hidden_state = self.layernorm_post(measurement_hidden_state)

            if [self.use_drawing_encoder, self.use_measurement_encoder, self.use_acronym_encoder].count(True) == 1:
                hidden_state = measurement_hidden_state
                output = measurement_ouput
        
        if self.use_drawing_encoder:
            drawing_output = self.drawing_encoder(hidden_state, attention_mask=attention_mask, output_hidden_states=True, output_attentions=output_attentions,)
            drawing_hidden_state = drawing_output[0]
            drawing_hidden_state = self.layernorm_post(drawing_hidden_state)
            
            if [self.use_drawing_encoder, self.use_measurement_encoder, self.use_acronym_encoder].count(True) == 1:
                hidden_state = drawing_hidden_state
                output = drawing_output
        
        # Feature-wise cross attention symmetrically 
        # Could clean up logic between cross attention and final addition
        if self.use_acronym_encoder and self.use_measurement_encoder:
            acronym_to_measurement, _ = self.acronym_to_measurement_Xattention(acronym_hidden_state , measurement_hidden_state)
            hidden_state = acronym_to_measurement
            output = [acronym_ouput, measurement_ouput]
        
        if self.use_acronym_encoder and self.use_drawing_encoder:
            acronym_to_drawing, _    = self.acronym_to_drawing_Xattention(acronym_hidden_state , drawing_hidden_state)
            hidden_state = acronym_to_drawing
            output = [acronym_ouput, drawing_output]
            
        if self.use_drawing_encoder and self.use_measurement_encoder:
            drawing_to_measurement, _ = self.drawing_to_measurement_Xattention(drawing_hidden_state , measurement_hidden_state)
            hidden_state = drawing_to_measurement
            output = [drawing_output, measurement_ouput]
            
        # add tensors across features to create enriched rep of the page if True on using all three encoders
        if self.use_drawing_encoder and self.use_measurement_encoder and self.use_acronym_encoder:
            hidden_state = acronym_to_measurement + drawing_to_measurement + acronym_to_drawing 
            output = [acronym_ouput, measurement_ouput, drawing_output]
        # print(f"After encoders {hidden_state.shape}") if curious else None
        
        # Apply global encoder , which will combine feature<>feature<>page -> BEST Rep
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media, num_tiles, num_patches + num_padding_patches, dim
        )
        hidden_state = self.post_tile_positional_embedding(hidden_state, aspect_ratio_ids)
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media, num_tiles * (num_patches + num_padding_patches), dim
        )
        global_output = self.global_transformer(
            hidden_state,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        hidden_state = global_output[0]
        # print(f"After global encoder {hidden_state.shape}") if curious else None
        
        # Remove padding form hidden state
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media, num_tiles, num_patches + num_padding_patches, dim
        )
        hidden_state = hidden_state[:, :, :slice_index]
        hidden_state = hidden_state.reshape(batch_size, num_concurrent_media, num_tiles, num_patches, dim)

        # Collect intermediate layer outputs from encoder output, if there's 1 and/or more encoders
        if [self.use_drawing_encoder, self.use_measurement_encoder, self.use_acronym_encoder].count(True) == 1:
            all_intermediate_hidden_states = [output[1][i] for i in self.intermediate_layers_indices]
        else:
            # could average instead of sum for better normalization 
            all_intermediate_hidden_states = [
                sum([indv_output[1][i] for indv_output in output]) for i in self.intermediate_layers_indices
                ]
            # torch.stack( [torch.tensor([1,2]) , torch.tensor([1,2])] , dim = -1)
        #
        #
        # MANUAL FIX - moving to the last gpu to prevent an overload
        # all_intermediate_hidden_states = [t.to("cuda:7") for t in all_intermediate_hidden_states]
        #
        #
        #
        
        intermediate_hidden_states = torch.stack(all_intermediate_hidden_states, dim=-1)
        
        # Remove padding from intermediate hidden states
        intermediate_hidden_states = intermediate_hidden_states.reshape(
            batch_size * num_concurrent_media, num_tiles, num_patches + num_padding_patches, -1
        )
        intermediate_hidden_states = intermediate_hidden_states[:, :, :slice_index]
        intermediate_hidden_states = intermediate_hidden_states.reshape(
            batch_size, num_concurrent_media, num_tiles, num_patches, -1
        )
        
        #
        #
        # MANUAL FIX FOR NOW 
        # hidden_state = hidden_state.to("cuda:7")
        # intermediate_hidden_states = intermediate_hidden_states.to("cuda:7")

        # print(hidden_state.device)
        # print(intermediate_hidden_states.device)
        #
        #
        #
        
        # Concatenate final hidden state and intermediate hidden states
        hidden_state = torch.cat([hidden_state, intermediate_hidden_states], dim=-1)

        if output_hidden_states:
            hidden_states = tuple(all_intermediate_hidden_states) + tuple(global_output[1])
        else:
            hidden_states = None

        if output_attentions:
            # global transformer in contrast to `self.transformer` doesn't always return hidden states so we might go index out-of-range
            global_attn = tuple(global_output[2]) if output_hidden_states else tuple(global_output[1])
            attentions = tuple(output[2]) + global_attn
        else:
            attentions = None

        if not return_dict:
            return tuple(v for v in [hidden_state, hidden_states, attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_state,
            hidden_states=hidden_states,
            attentions=attentions,
        )


'''
Replace vision model w/ the SuperVisionModel class so that newly intialized weights 
and their names are used instead of the base Mllama initialization.
'''
class MllamaProjectManagerForConditionalGeneration(MllamaForConditionalGeneration, GenerationMixin):

    def __init__(self, config:MllamaConfig):
        super().__init__(config)
        self.vision_model = MllamaSuperVisionModel._from_config(config.vision_config)


#
