"""Checkpoint-compatible model families used in the completed UIEB studies."""

from .baselines import EnhancementModel
from .conditioning import ConditionedUNet
from .global_latent import GlobalLatentUNet


def build_model(config, condition, seed):
    if condition not in config["conditions"] or seed not in config["seeds"]:
        raise ValueError("Condition/seed is outside this experiment")
    family = config["model_family"]
    if family == "baseline":
        return EnhancementModel(config["backbone"], condition, seed)
    if family == "conditioning":
        return ConditionedUNet(condition, seed)
    if family == "global":
        return GlobalLatentUNet(int(condition[1:]), seed)
    raise ValueError(family)
