
from torch import nn

'''
Sets embedding layers for full parameter finetuning, if reverse then will revert 
all params that are not in the embedding layers to requires_grad=False. 

Custom function for the model that you are working w/ and the layers you want it to target.
'''
def prepare_vision_embedding_layers(
    model:nn.Module,
    reverse:bool=False
):
    embeddings = [
        "vision_model.gated_positional_embedding.embedding",
        "vision_model.gated_positional_embedding.tile_embedding",
        "vision_model.patch_embedding",
        "vision_model.post_tile_positional_embedding.embedding",
        "vision_model.pre_tile_positional_embedding.embedding",
    ]

    for name, param in model.named_parameters():
        match = any(embed in name for embed in embeddings)
        param.requires_grad = not match if reverse else match


def prepare_vision_layers(
    model:nn.Module
):
    for name,param in model.vision_model.named_parameters():
        param.requires_grad=True


if __name__ == "__main__":
    None


#
