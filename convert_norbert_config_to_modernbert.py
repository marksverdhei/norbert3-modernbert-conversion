#!/usr/bin/env python
"""
A conversion script for merging a NorBERT (norbert3) config with a ModernBERT config.

Every applicable parameter from NorBERT is used to override the corresponding ModernBERT setting.
Other parameters (e.g. token IDs, classifier options, etc.) are taken from the ModernBERT config.

Usage:
    python convert_norbert_config_to_modernbert.py --norbert_config PATH_TO_NORBERT_CONFIG.json \
                                                   --modernbert_config PATH_TO_MODERNBERT_CONFIG.json \
                                                   --output_config PATH_TO_OUTPUT_CONFIG.json
"""

import json
import copy
import smart_open

def convert_config(norbert_config, modernbert_config):
    """
    Merge the NorBERT configuration into the ModernBERT configuration.
    
    Parameters from NorBERT are mapped/renamed as follows:
      - "attention_probs_dropout_prob"  -> "attention_dropout"
      - "hidden_dropout_prob"           -> "mlp_dropout"
      - "layer_norm_eps"                -> "norm_eps"
      - "position_bucket_size"          -> (added) and "position_embedding_type" is set to "relative"
      - Model dimensions ("hidden_size", "intermediate_size", "max_position_embeddings",
         "num_attention_heads", "num_hidden_layers", "vocab_size") override ModernBERT's.
    
    The "architectures" key is replaced with ModernBERT’s expected architecture.
    The "auto_map" key is dropped.
    
    Args:
        norbert_config (dict): The configuration dictionary from NorBERT.
        modernbert_config (dict): The ModernBERT configuration dictionary.
    
    Returns:
        dict: A new configuration dictionary in ModernBERT format.
    """
    # Start with a deep copy of the ModernBERT config.
    new_config = copy.deepcopy(modernbert_config)

    # --- Override with applicable NorBERT parameters ---

    # Dropout parameters
    if "attention_probs_dropout_prob" in norbert_config:
        new_config["attention_dropout"] = norbert_config["attention_probs_dropout_prob"]
    if "hidden_dropout_prob" in norbert_config:
        new_config["mlp_dropout"] = norbert_config["hidden_dropout_prob"]

    # Model dimensions
    for key in ["hidden_size", "intermediate_size", "max_position_embeddings", "num_attention_heads", "num_hidden_layers", "vocab_size"]:
        if key in norbert_config:
            new_config[key] = norbert_config[key]

    # Layer norm epsilon (ModernBERT uses "norm_eps")
    if "layer_norm_eps" in norbert_config:
        new_config["norm_eps"] = norbert_config["layer_norm_eps"]

    # Torch data type (if provided)
    if "torch_dtype" in norbert_config:
        new_config["torch_dtype"] = norbert_config["torch_dtype"]

    # Positional embeddings:
    # NorBERT provides "position_bucket_size" (for relative positions). If present, add it and
    # switch the embedding type to "relative".
    if "position_bucket_size" in norbert_config:
        new_config["position_bucket_size"] = norbert_config["position_bucket_size"]
        new_config["position_embedding_type"] = "relative"
    
    # --- Override keys that are model-specific ---

    # Use ModernBERT's architecture naming. (We disregard NorBERT's "architectures" value.)
    new_config["architectures"] = ["ModernBertForMaskedLM"]

    # Remove keys that are not applicable in ModernBERT
    if "auto_map" in new_config:
        del new_config["auto_map"]

    return new_config

def main():


    # Load NorBERT config.
    with smart_open.open("https://huggingface.co/ltg/norbert3-base/raw/main/config.json", "r", encoding="utf-8") as f:
        norbert_config = json.load(f)

    # Load ModernBERT config.
    with smart_open.open("https://huggingface.co/answerdotai/ModernBERT-base/raw/main/config.json", "r", encoding="utf-8") as f:
        modernbert_config = json.load(f)

    # Merge the configs.
    new_config = convert_config(norbert_config, modernbert_config)

    # Save the new configuration.
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(new_config, f, indent=2)
    print(f"Converted config saved to output.json")

if __name__ == "__main__":
    main()
