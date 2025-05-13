

from typing import Optional
from sentence_transformers import SentenceTransformer


from .configs import EmbedConfig


def build_embed_model(
    env:dict,
    embed_cfg:EmbedConfig,
    device:str,
):
    """
    Builds sentence embedding model from the source if an instance of it isn't avaliable in the current enviornment.
    Goal is to save gpu memory by preventing multiple instances of the same model from being instantiated.
    """

    if not isinstance(env, dict):
        raise TypeError(
            f"env is suppose to be a dictionary from the working environment. Is of type: {type(env)}"
        )
    MODEL="model"
    for var_name,value in env.items():
        if hasattr(value, MODEL):
            model=getattr(value,MODEL)
            if isinstance( model , SentenceTransformer):
                print(f"Fetched embed model from globals.")
                return model
    print(
        f"No instance of embed model found, building from source."
    )
    model=SentenceTransformer(
        embed_cfg.model_name,
        device=device
    )
    return model






#