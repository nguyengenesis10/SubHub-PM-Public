
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPast, CausalLMOutputWithPast
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
from transformers.models.mllama.modeling_mllama import (
    _prepare_aspect_ratio_attention_mask,
    _prepare_cross_attention_mask,
)

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from torch import nn


class MllamaPrecomputedAspectRatioEmbedding(nn.Module):
    def __init__(self, config: MllamaVisionConfig, is_gated: bool = True):
        super().__init__()
        self.max_num_tiles = config.max_num_tiles
        self.hidden_size = config.hidden_size
        self.max_aspect_ratio_id = config.max_aspect_ratio_id
        self.is_gated = is_gated

        self.embedding = nn.Embedding(self.max_aspect_ratio_id + 1, self.max_num_tiles * self.hidden_size)
        if is_gated:
            self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_state: torch.Tensor, aspect_ratio_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(aspect_ratio_ids)
        embeddings = embeddings.reshape(-1, self.max_num_tiles, 1, self.hidden_size)

        if self.is_gated:
            embeddings = embeddings * self.gate.tanh()

        hidden_state = hidden_state + embeddings
        return hidden_state


class MllamaPrecomputedPositionEmbedding(nn.Module):
    def __init__(self, config: MllamaVisionConfig):
        super().__init__()
        self.max_num_tiles = config.max_num_tiles
        self.max_aspect_ratio_id = config.max_aspect_ratio_id
        self.num_patches = (config.image_size // config.patch_size) ** 2 + 1
        self.hidden_size = config.hidden_size
        self.scale = config.hidden_size**-0.5

        self.gate = nn.Parameter(torch.zeros(1))

        # position embedding
        position_embedding = torch.randn(self.num_patches, self.hidden_size)
        self.embedding = nn.Parameter(self.scale * position_embedding)

        # tile position embedding
        self.tile_embedding = nn.Embedding(
            self.max_aspect_ratio_id + 1, self.max_num_tiles * self.num_patches * self.hidden_size
        )

    def forward(self, hidden_state: torch.Tensor, aspect_ratio_ids: torch.Tensor) -> torch.Tensor:
        # position embeddings
        gated_position_embedding = (1 - self.gate.tanh()) * self.embedding
        hidden_state = hidden_state + gated_position_embedding.view(1, 1, self.num_patches, self.hidden_size)

        # precomputed tile position embeddings
        tile_position_embedding = self.tile_embedding(aspect_ratio_ids)
        batch_size = hidden_state.shape[0]
        tile_position_embedding = tile_position_embedding.reshape(
            batch_size, self.max_num_tiles, self.num_patches, self.hidden_size
        )
        gated_tile_position_embedding = self.gate.tanh() * tile_position_embedding
        hidden_state = hidden_state + gated_tile_position_embedding

        return hidden_state


# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->MllamaVision
class MllamaVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class MllamaVisionAttention(nn.Module):
    def __init__(self, config: MllamaVisionConfig):
        super().__init__()

        self.embed_dim = config.hidden_size
        self.num_heads = config.attention_heads
        # floor division, auto rounds down
        self.head_dim = config.hidden_size // config.attention_heads

        self.q_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.embed_dim, bias=False)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = None,
    ) -> torch.Tensor:
        query = self.q_proj(hidden_state)
        key   = self.k_proj(hidden_state)
        value = self.v_proj(hidden_state)

        batch_size, q_seq_len, _ = query.shape
        _, kv_seq_len, _ = key.shape

        query = query.view(batch_size, q_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key   = key.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_output = torch.matmul(attn_weights, value)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_seq_len, -1)

        output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return output, attn_weights


class MllamaVisionSdpaAttention(MllamaVisionAttention):
    # Adapted from MllamaVisionAttention
    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = None,
    ) -> torch.Tensor:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        if output_attentions:
            logger.warning_once(
                "MllamaModel is using MllamaVisionSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_state=hidden_state,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )

        query = self.q_proj(hidden_state)
        key = self.k_proj(hidden_state)
        value = self.v_proj(hidden_state)

        batch_size, q_seq_len, _ = query.shape
        _, kv_seq_len, _ = key.shape

        query = query.view(batch_size, q_seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, kv_seq_len, self.num_heads, self.head_dim)
        value = value.view(batch_size, kv_seq_len, self.num_heads, self.head_dim)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_seq_len, -1)

        output = self.o_proj(attn_output)

        return output, None

'''
Cross attention between features on the page , no need for an o_proj b/c we're just computing a single head of attn, 
it's a new feature.
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



MLLAMA_VISION_ATTENTION_CLASSES = {"eager": MllamaVisionAttention, "sdpa": MllamaVisionSdpaAttention}


class MllamaVisionEncoderLayer(nn.Module):
    def __init__(self, config: MllamaVisionConfig, is_gated: bool = False):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.attention_heads
        self.is_gated = is_gated
        self.intermediate_size = config.intermediate_size

        self.self_attn = MLLAMA_VISION_ATTENTION_CLASSES[config._attn_implementation](config)
        self.mlp = MllamaVisionMLP(config)

        self.input_layernorm = nn.LayerNorm(self.hidden_size, eps=config.norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(self.hidden_size, eps=config.norm_eps)

        if is_gated:
            self.gate_attn = nn.Parameter(torch.ones(1) * math.pi / 4)
            self.gate_ffn = nn.Parameter(torch.ones(1) * math.pi / 4)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = None,
    ):
        # Self Attention
        residual = hidden_state
        hidden_state = self.input_layernorm(hidden_state)
        hidden_state, attn_weights = self.self_attn(hidden_state, attention_mask=attention_mask)
        if self.is_gated:
            hidden_state = self.gate_attn.tanh() * hidden_state
        hidden_state = residual + hidden_state

        # Feed forward
        residual = hidden_state
        hidden_state = self.post_attention_layernorm(hidden_state)
        hidden_state = self.mlp(hidden_state)
        #print(self.is_gated)
        if self.is_gated:
            hidden_state = self.gate_ffn.tanh() * hidden_state
        hidden_state = residual + hidden_state

        outputs = (hidden_state,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class MllamaVisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`MllamaEncoderLayer`].

    Args:
        config: MllamaConfig
    """

    def __init__(self, config: MllamaVisionConfig, num_layers=32, is_gated=False):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([MllamaVisionEncoderLayer(config, is_gated) for _ in range(num_layers)])
        self.gradient_checkpointing = False
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_state=hidden_states,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                )

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

            hidden_states = layer_outputs[0]

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class MllamaPreTrainedModel(PreTrainedModel):
    config_class = MllamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "MllamaVisionEncoderLayer",
        "MllamaCrossAttentionDecoderLayer",
        "MllamaSelfAttentionDecoderLayer",
    ]
    _supports_cache_class = True
    _supports_static_cache = False  # static cache cannot have different shapes for each layer
    _supports_sdpa = True
    _supports_quantized_cache = True

    def _init_weights(self, module):
        std = self.config.get_text_config().initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.Parameter):
            module.data.normal_(mean=0.0, std=std)
        elif isinstance(module, MllamaVisionModel):
            nn.init.normal_(module.class_embedding.data, std=std)
        elif isinstance(module, MllamaPrecomputedPositionEmbedding):
            nn.init.normal_(module.embedding.data, std=std)
        elif isinstance(module, MllamaVisionEncoderLayer) and module.is_gated:
            nn.init.normal_(module.gate_attn.data, std=std)
            nn.init.normal_(module.gate_ffn.data, std=std)

    # Copied from transformers.models.llama.modeling_llama.LlamaModel._update_causal_mask
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask
    
    
    @staticmethod
    # Copied from transformers.models.llama.modeling_llama.LlamaModel._prepare_4d_causal_attention_mask_with_cache_position
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


# get context within a page
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

        Returns:

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
from transformers.models.mllama.modeling_mllama_vision import MllamaSuperVisionModel, MllamaVisionModel, SuperEncoders
from transformers.models.mllama.configuration_mllama import MllamaVisionConfig

model_path = "/Users/genesisnguyen/llama3.2-11B-vision-instruct"

vision_config = MllamaVisionConfig.from_pretrained(
    model_path, 
    use_acronym_encoder = True,
    use_measurement_encoder = True, 
    use_drawing_encoder = True,
)

print(vision_config.use_acronym_encoder) ; print(vision_config.use_measurement_encoder) ; print(vision_config.use_drawing_encoder)

model = MllamaSuperVisionModel.from_pretrained(model_path, config = vision_config, ignore_mismatched_sizes = True)


# copies weights from base model to encode layer, so we don't have to train from scratch
model.prepare_custom_finetuning()


model_1 = MllamaVisionModel.from_pretrained(model_path)


model_1.transformer.layers[0].self_attn.q_proj.weight == model.acronym_encoder.layers[0].self_attn.q_proj.weight


'''

'''
from huggingface_hub import login
from datasets import load_dataset
from llama_recipes.datasets.id_trade_dataset import tokenize_dialogs
from transformers import AutoProcessor
import ast

model_path = "/Users/genesisnguyen/llama3.2-11B-vision-instruct"

login(token="[REMOVED_SECRET]" , add_to_git_credential=True)
dataset = load_dataset( "genesis1SubHub/WP_AFB_PG16")

dialogs_train = [ast.literal_eval(i) for i in dataset["train"]["texts"]]
images_train  = dataset["train"]["images"]

processor = AutoProcessor.from_pretrained(model_path)

batch = tokenize_dialogs(dialogs_train, images_train, processor)

msl = len(max(batch.keys() , key = len))

for i in batch.keys():
    # only print shapes for the first sample of each key
    for sample in batch[i][:1]:
        print(f"Key: {i}{' ' * (msl - len(i))}" , sample.shape)

'''

# freeze all model layers
def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


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
        
        self.use_acronym_encoder     = config.use_acronym_encoder
        self.use_measurement_encoder = config.use_measurement_encoder
        self.use_drawing_encoder     = config.use_drawing_encoder
        
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
        
        if [self.use_acronym_encoder , self.use_measurement_encoder,self.use_drawing_encoder].count(True) == 1:
            # somewhat wasteful, but the base model is only 900M params, I figure it should be fine
            model = MllamaVisionModel.from_pretrained(self.config.pretrained_model_name_or_path)
            
            print(f"Copying weights from {self.config.pretrained_model_name_or_path}...")
            
            if self.use_acronym_encoder:
                for layer , base_model_layer in zip(self.acronym_encoder.layers ,model.transformer.layers):
                    # self-attention copy
                    layer.self_attn.q_proj.weight.data = base_model_layer.self_attn.q_proj.weight.data.clone()
                    layer.self_attn.k_proj.weight.data = base_model_layer.self_attn.k_proj.weight.data.clone()
                    layer.self_attn.v_proj.weight.data = base_model_layer.self_attn.v_proj.weight.data.clone()
                    layer.self_attn.o_proj.weight.data = base_model_layer.self_attn.o_proj.weight.data.clone()
                    
                    # multi-layer-perceptron copy
                    layer.mlp.fc1.weight.data = base_model_layer.mlp.fc1.weight.data.clone()
                    layer.mlp.fc1.bias.data = base_model_layer.mlp.fc1.bias.data.clone()

                    layer.mlp.fc2.weight.data = base_model_layer.mlp.fc2.weight.data.clone()
                    layer.mlp.fc2.bias.data = base_model_layer.mlp.fc2.bias.data.clone()
                    
                    # normalization copy
                    layer.input_layernorm.weight.data = base_model_layer.input_layernorm.weight.data.clone()
                    layer.post_attention_layernorm.weight.data = base_model_layer.post_attention_layernorm.weight.data.clone()
                
                
                print(f"Finished copying weights to the acronym_encoder.")
                
            if self.use_measurement_encoder:
                for layer , base_model_layer in zip(self.measurement_encoder.layers ,model.transformer.layers):
                    
                    # self-attention copy
                    layer.self_attn.q_proj.weight.data = base_model_layer.self_attn.q_proj.weight.data.clone()
                    layer.self_attn.k_proj.weight.data = base_model_layer.self_attn.k_proj.weight.data.clone()
                    layer.self_attn.v_proj.weight.data = base_model_layer.self_attn.v_proj.weight.data.clone()
                    layer.self_attn.o_proj.weight.data = base_model_layer.self_attn.o_proj.weight.data.clone()
                    
                    # multi-layer-perceptron copy
                    layer.mlp.fc1.weight.data = base_model_layer.mlp.fc1.weight.data.clone()
                    layer.mlp.fc1.bias.data = base_model_layer.mlp.fc1.bias.data.clone()

                    layer.mlp.fc2.weight.data = base_model_layer.mlp.fc2.weight.data.clone()
                    layer.mlp.fc2.bias.data = base_model_layer.mlp.fc2.bias.data.clone()
                    
                    # normalization copy
                    layer.input_layernorm.weight.data = base_model_layer.input_layernorm.weight.data.clone()
                    layer.post_attention_layernorm.weight.data = base_model_layer.post_attention_layernorm.weight.data.clone()
            
                print(f"Finished copying weights to the measurement_encoder.")
            
            if self.use_drawing_encoder:
                for layer , base_model_layer in zip(self.drawing_encoder.layers ,model.transformer.layers):
                    
                    # self-attention copy
                    layer.self_attn.q_proj.weight.data = base_model_layer.self_attn.q_proj.weight.data.clone()
                    layer.self_attn.k_proj.weight.data = base_model_layer.self_attn.k_proj.weight.data.clone()
                    layer.self_attn.v_proj.weight.data = base_model_layer.self_attn.v_proj.weight.data.clone()
                    layer.self_attn.o_proj.weight.data = base_model_layer.self_attn.o_proj.weight.data.clone()
                    
                    # multi-layer-perceptron copy
                    layer.mlp.fc1.weight.data = base_model_layer.mlp.fc1.weight.data.clone()
                    layer.mlp.fc1.bias.data = base_model_layer.mlp.fc1.bias.data.clone()

                    layer.mlp.fc2.weight.data = base_model_layer.mlp.fc2.weight.data.clone()
                    layer.mlp.fc2.bias.data = base_model_layer.mlp.fc2.bias.data.clone()
                    
                    # normalization copy
                    layer.input_layernorm.weight.data = base_model_layer.input_layernorm.weight.data.clone()
                    layer.post_attention_layernorm.weight.data = base_model_layer.post_attention_layernorm.weight.data.clone()
            
                print(f"Finished copying weights to the drawing_encoder.")
    
    
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


#### Custom Class w/ only difference being the replacement of the SuperVisionModel Architecture config
class MllamaProjectManagerForConditionalGeneration(MllamaPreTrainedModel, GenerationMixin):
    _supports_quantized_cache = False  # quant cache not supported in encoder-decoder setting

    def __init__(self, config: MllamaConfig):
        super().__init__(config)
        self.vocab_size = config.text_config.vocab_size
        self.hidden_size = config.text_config.hidden_size
        self.max_num_tiles = config.vision_config.max_num_tiles
        self.vision_output_dim = config.vision_config.vision_output_dim
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

        self.vision_model = MllamaSuperVisionModel._from_config(config.vision_config)
        self.language_model = MllamaForCausalLM._from_config(config.text_config)
        self.multi_modal_projector = nn.Linear(
            config.vision_config.vision_output_dim,
            config.text_config.hidden_size,
            bias=True,
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    # @add_start_docstrings_to_model_forward(MLLAMA_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class="MllamaConfig")
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        aspect_ratio_mask: Optional[torch.Tensor] = None,
        aspect_ratio_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_states: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.


        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, MllamaForConditionalGeneration

        >>> checkpoint = "meta-llama/Llama-3.2-11B-Vision"
        >>> model = MllamaForConditionalGeneration.from_pretrained(checkpoint)
        >>> processor = AutoProcessor.from_pretrained(checkpoint)

        >>> prompt = "<|image|>If I had to write a haiku for this one"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

        >>> # Generate
        >>> output = model.generate(**inputs, max_new_tokens=15)

        >>> prompt_len = inputs.input_ids.shape[-1]
        >>> generated_ids = output[:, prompt_len:]
        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        >>> print(generated_text)
        [', it would be:.\\nA stop sign in Chinatown.\\n']
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if pixel_values is not None and cross_attention_states is not None:
            raise ValueError("`pixel_values` and `cross_attention_states` cannot be provided simultaneously")

        if pixel_values is not None:
            if aspect_ratio_ids is None:
                raise ValueError("`aspect_ratio_ids` must be provided if `pixel_values` is provided")
            # get vision tokens from vision model
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                aspect_ratio_ids=aspect_ratio_ids,
                aspect_ratio_mask=aspect_ratio_mask,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                return_dict=return_dict,
            )
            cross_attention_states = vision_outputs[0]
            cross_attention_states = self.multi_modal_projector(cross_attention_states).reshape(
                -1, cross_attention_states.shape[-2], self.hidden_size
            )

        if cross_attention_mask is not None:
            cross_attention_mask, full_text_row_masked_out_mask = _prepare_cross_attention_mask(
                cross_attention_mask,
                num_vision_tokens=self.vision_model.num_patches,
                dtype=self.dtype,
            )
        else:
            full_text_row_masked_out_mask = None

        if cross_attention_mask is not None and cache_position is not None:
            cross_attention_mask = cross_attention_mask[:, :, cache_position]
            full_text_row_masked_out_mask = full_text_row_masked_out_mask[:, :, cache_position]

        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cross_attention_states=cross_attention_states,
            cross_attention_mask=cross_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
        )

        return outputs

    def prepare_inputs_for_generation(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        pixel_values=None,
        aspect_ratio_ids=None,
        aspect_ratio_mask=None,
        cross_attention_mask=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        num_logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        # TODO: we have no attention_mask so this won't work, check if we really won't need attention mask and find another way
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "cross_attention_mask": cross_attention_mask,
            }
        )

        # If we're in pre-fill or cacheless decoding step, then we need pixel_values and aspect ratios
        # to compute image hidden states, otherwise they are cached within each cross attn layer
        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["aspect_ratio_ids"] = aspect_ratio_ids
            model_inputs["aspect_ratio_mask"] = aspect_ratio_mask

        return model_inputs

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder, **kwargs):
        cross_attention_mask_prev = model_kwargs.get("cross_attention_mask", None)
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )

        # add cross-attn mask for new token
        if cross_attention_mask_prev is not None:
            model_kwargs["cross_attention_mask"] = torch.cat(
                [cross_attention_mask_prev, cross_attention_mask_prev[:, -1:, ...]], dim=1
            )
        return model_kwargs







#
