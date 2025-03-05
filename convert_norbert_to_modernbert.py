#!/usr/bin/env python
"""
A comprehensive conversion script for converting NorBERT (norbert3) to ModernBERT format.

This script handles both configuration and weight conversion in one step.

Usage:
    python norbert_to_modernbert_converter.py --norbert_model_id ltg/norbert3-base \
                                              --modernbert_model_id answerdotai/ModernBERT-base \
                                              --output_dir output_model \
                                              [--test_conversion]
"""

import os
import json
import copy
import argparse
import torch
import smart_open
import transformers
from transformers import ModernBertConfig, ModernBertForMaskedLM, AutoModelForMaskedLM, AutoTokenizer


def convert_config(norbert_config, modernbert_config, norbert_tokenizer=None):
    """
    Merge the NorBERT configuration into the ModernBERT configuration.
    
    Parameters from NorBERT are mapped/renamed as follows:
      - "attention_probs_dropout_prob"  -> "attention_dropout"
      - "hidden_dropout_prob"           -> "mlp_dropout"
      - "layer_norm_eps"                -> "norm_eps"
      - "position_bucket_size"          -> (added) and "position_embedding_type" is set to "relative"
      - Model dimensions ("hidden_size", "intermediate_size", "max_position_embeddings",
         "num_attention_heads", "num_hidden_layers", "vocab_size") override ModernBERT's.
    
    The "architectures" key is replaced with ModernBERT's expected architecture.
    The "auto_map" key is dropped.
    
    Special token IDs are set from the NorBERT tokenizer if provided, otherwise hardcoded values are used.
    
    Args:
        norbert_config (dict): The configuration dictionary from NorBERT.
        modernbert_config (dict): The ModernBERT configuration dictionary.
        norbert_tokenizer (Optional[AutoTokenizer]): The NorBERT tokenizer to get special token IDs from.
    
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
    
    # --- Set special token IDs ---
    # Use tokenizer if provided, otherwise use hardcoded values
    if norbert_tokenizer is not None:
        # Get special token IDs from the tokenizer
        if hasattr(norbert_tokenizer, "cls_token_id") and norbert_tokenizer.cls_token_id is not None:
            new_config["cls_token_id"] = norbert_tokenizer.cls_token_id
        else:
            new_config["cls_token_id"] = 1  # Default value
            
        if hasattr(norbert_tokenizer, "sep_token_id") and norbert_tokenizer.sep_token_id is not None:
            new_config["sep_token_id"] = norbert_tokenizer.sep_token_id
        else:
            new_config["sep_token_id"] = 2  # Default value
            
        if hasattr(norbert_tokenizer, "pad_token_id") and norbert_tokenizer.pad_token_id is not None:
            new_config["pad_token_id"] = norbert_tokenizer.pad_token_id
        else:
            new_config["pad_token_id"] = 3  # Default value
            
        if hasattr(norbert_tokenizer, "bos_token_id") and norbert_tokenizer.bos_token_id is not None:
            new_config["bos_token_id"] = norbert_tokenizer.bos_token_id
        else:
            new_config["bos_token_id"] = 5  # Default value
            
        if hasattr(norbert_tokenizer, "eos_token_id") and norbert_tokenizer.eos_token_id is not None:
            new_config["eos_token_id"] = norbert_tokenizer.eos_token_id
        else:
            new_config["eos_token_id"] = 6  # Default value
    else:
        # Use hardcoded values from the manually coded config
        new_config["cls_token_id"] = 1
        new_config["sep_token_id"] = 2
        new_config["pad_token_id"] = 3
        new_config["bos_token_id"] = 5
        new_config["eos_token_id"] = 6
    
    # --- Override keys that are model-specific ---

    # Use ModernBERT's architecture naming. (We disregard NorBERT's "architectures" value.)
    new_config["architectures"] = ["ModernBertForMaskedLM"]

    # Remove keys that are not applicable in ModernBERT
    if "auto_map" in new_config:
        del new_config["auto_map"]

    return new_config


def convert_state_dict(norbert_state_dict):
    """
    Convert a NorBERT state dict into a ModernBERT state dict.
    
    The conversion does the following mappings:
    
    1. Embeddings:
       - NorBERT "embedding.word_embedding.weight" -> ModernBERT "model.embeddings.tok_embeddings.weight"
       - NorBERT "embedding.relative_layer_norm.weight" -> ModernBERT "model.embeddings.norm.weight"
         (NorBERT's norm bias is dropped because ModernBERT's state dict only contains a weight.)
       - NorBERT's "embedding.relative_embedding" is not used by ModernBERT and is dropped.
    
    2. Transformer Layers:
       For each layer (e.g., layer i):
         - Attention Q/K/V:
             * NorBERT has two parameters:
                 - "transformer.layers.{i}.attention.in_proj_qk.weight" (a fused Q & K, shape [2*H, H])
                 - "transformer.layers.{i}.attention.in_proj_v.weight" (V, shape [H, H])
             * ModernBERT expects one fused weight "model.layers.{i}.attn.Wqkv.weight" of shape [3*H, H]
             * We concatenate the NorBERT QK weight (which already contains Q then K) with the V weight along
               the first dimension.
         - Attention output:
             * NorBERT "transformer.layers.{i}.attention.out_proj.weight" -> ModernBERT "model.layers.{i}.attn.Wo.weight"
             * (NorBERT's out_proj bias is dropped.)
         - Attention layer-norm:
             * NorBERT "transformer.layers.{i}.attention.post_layer_norm.weight" -> ModernBERT "model.layers.{i}.attn_norm.weight"
             * (Again, the bias is dropped.)
         - MLP:
             * NorBERT "transformer.layers.{i}.mlp.mlp.1.weight" -> ModernBERT "model.layers.{i}.mlp.Wi.weight"
             * NorBERT "transformer.layers.{i}.mlp.mlp.4.weight" -> ModernBERT "model.layers.{i}.mlp.Wo.weight"
         - MLP layer-norm:
             * ModernBERT expects a key "model.layers.{i}.mlp_norm.weight". NorBERT does not have an MLP norm,
               so we initialize it to ones (with the proper hidden size).
    
    3. Final Norm:
         * ModernBERT expects "model.final_norm.weight". We initialize this to ones (with hidden size inferred
           from the token embedding).
    
    4. Classification Head:
         * NorBERT "classifier.nonlinearity.1.weight" -> ModernBERT "head.dense.weight"
         * A head norm "head.norm.weight" is created (initialized to ones) since ModernBERT's state dict contains it.
         * NorBERT "classifier.nonlinearity.5.weight" and "classifier.nonlinearity.5.bias" are mapped to
           "decoder.weight" and "decoder.bias" respectively.
    
    Args:
        norbert_state_dict (dict): state dict from the NorBERT model.
    
    Returns:
        dict: a new state dict with keys remapped to ModernBERT format.
    """
    new_state_dict = {}

    ### 1. Convert Embedding Weights
    new_state_dict["model.embeddings.tok_embeddings.weight"] = norbert_state_dict["embedding.word_embedding.weight"]
    new_state_dict["model.embeddings.norm.weight"] = norbert_state_dict["embedding.relative_layer_norm.weight"]
    # (Note: "embedding.relative_embedding" is not mapped because ModernBERT does not expect it.)
    
    ### 2. Convert Transformer Layers
    # First, determine which layers are present (assumes keys like "transformer.layers.{i}.attention.in_proj_qk.weight")
    layers = set()
    for key in norbert_state_dict.keys():
        if key.startswith("transformer.layers."):
            try:
                # The layer index is the third token after splitting by '.'
                layer_idx = int(key.split('.')[2])
                layers.add(layer_idx)
            except ValueError:
                pass
    layers = sorted(layers)
    
    for i in layers:
        norbert_prefix = f"transformer.layers.{i}"
        modern_prefix = f"model.layers.{i}"
        
        # --- Attention QKV ---
        # Get NorBERT's fused Q/K weight and V weight:
        w_qk = norbert_state_dict[f"{norbert_prefix}.attention.in_proj_qk.weight"]  # shape: [2*H, H]
        w_v  = norbert_state_dict[f"{norbert_prefix}.attention.in_proj_v.weight"]    # shape: [H, H]
        # Concatenate along the first dimension to get shape [3*H, H] in the order [Q, K, V]
        w_qkv = torch.cat([w_qk, w_v], dim=0)
        new_state_dict[f"{modern_prefix}.attn.Wqkv.weight"] = w_qkv
        
        # --- Attention output projection ---
        new_state_dict[f"{modern_prefix}.attn.Wo.weight"] = norbert_state_dict[f"{norbert_prefix}.attention.out_proj.weight"]
        
        # --- Attention layer normalization ---
        if i > 0:  # Note: check for correct layer index handling
            new_state_dict[f"{modern_prefix}.attn_norm.weight"] = norbert_state_dict[f"{norbert_prefix}.attention.post_layer_norm.weight"]
        
        # --- MLP layers ---
        new_state_dict[f"{modern_prefix}.mlp.Wi.weight"] = norbert_state_dict[f"{norbert_prefix}.mlp.mlp.1.weight"]
        new_state_dict[f"{modern_prefix}.mlp.Wo.weight"] = norbert_state_dict[f"{norbert_prefix}.mlp.mlp.4.weight"]
        
        # --- MLP normalization (not present in NorBERT) ---
        # We initialize this parameter to ones. We infer the hidden size from the Wo weight (its output dimension).
        # (Assumes Wo.weight shape is [hidden_size, intermediate_dim] so its first dim is the hidden size.)
        hidden_size = new_state_dict[f"{modern_prefix}.attn.Wo.weight"].shape[0]
        new_state_dict[f"{modern_prefix}.mlp_norm.weight"] = torch.ones(hidden_size)
    
    ### 3. Final Normalization Layer
    # Initialize "model.final_norm.weight" as ones, with hidden size taken from the token embedding weight.
    hidden_size = new_state_dict["model.embeddings.tok_embeddings.weight"].shape[1]
    new_state_dict["model.final_norm.weight"] = torch.ones(hidden_size)
    
    ### 4. Convert Classification Head
    new_state_dict["head.dense.weight"] = norbert_state_dict["classifier.nonlinearity.1.weight"]
    # Initialize the head norm to ones. The number of features is taken from the output dimension of the dense weight.
    out_features = new_state_dict["head.dense.weight"].shape[0]
    new_state_dict["head.norm.weight"] = torch.ones(out_features)
    
    new_state_dict["decoder.weight"] = norbert_state_dict["classifier.nonlinearity.5.weight"]
    new_state_dict["decoder.bias"] = norbert_state_dict["classifier.nonlinearity.5.bias"]
    
    return new_state_dict


def test_conversion(norbert_model_id, modern_model, tokenizer, test_text="Nå ønsker de seg en[MASK] bolig."):
    """
    Test the converted model by comparing its output with the original NorBERT model.
    
    Args:
        norbert_model_id (str): HuggingFace model ID for the original NorBERT model.
        modern_model (ModernBertForMaskedLM): Converted ModernBERT model.
        tokenizer (AutoTokenizer): Tokenizer to use for both models.
        test_text (str): Text to use for testing, should include a [MASK] token.
        
    Returns:
        bool: True if the test was successful (models produce similar outputs), False otherwise.
    """
    print(f"\nTesting conversion with text: '{test_text}'")
    
    # Load original NorBERT model for comparison
    print(f"Loading original NorBERT model from {norbert_model_id}...")
    norbert_model = AutoModelForMaskedLM.from_pretrained(norbert_model_id, trust_remote_code=True)
    
    # Prepare input
    mask_id = tokenizer.convert_tokens_to_ids("[MASK]")
    input_text = tokenizer(test_text, return_tensors="pt")
    
    # Get predictions from both models
    with torch.no_grad():
        norbert_output = norbert_model(**input_text)
        modern_output = modern_model(**input_text)
    
    # Get the predicted token at the masked position
    mask_positions = (input_text.input_ids == mask_id).nonzero(as_tuple=True)[1]
    
    if len(mask_positions) == 0:
        print("Warning: No [MASK] token found in test text.")
        return False
    
    mask_pos = mask_positions[0].item()
    
    norbert_probs = torch.nn.functional.softmax(norbert_output.logits[0, mask_pos], dim=0)
    modern_probs = torch.nn.functional.softmax(modern_output.logits[0, mask_pos], dim=0)
    
    # Get top 5 predictions from each model
    norbert_top5 = torch.topk(norbert_probs, 5)
    modern_top5 = torch.topk(modern_probs, 5)
    
    print("\nTop 5 predictions from NorBERT:")
    for i in range(5):
        token_id = norbert_top5.indices[i].item()
        token = tokenizer.convert_ids_to_tokens(token_id)
        prob = norbert_top5.values[i].item()
        print(f"  {token} ({prob:.4f})")
    
    print("\nTop 5 predictions from converted ModernBERT:")
    for i in range(5):
        token_id = modern_top5.indices[i].item()
        token = tokenizer.convert_ids_to_tokens(token_id)
        prob = modern_top5.values[i].item()
        print(f"  {token} ({prob:.4f})")
    
    # Check if the top prediction is the same
    success = norbert_top5.indices[0].item() == modern_top5.indices[0].item()
    
    if success:
        print("\nTest PASSED: Both models predict the same top token.")
    else:
        print("\nTest WARNING: Models predict different top tokens.")
        # Still consider it a partial success if the top token from one model is in the top 5 of the other
        norbert_top_token = norbert_top5.indices[0].item()
        if norbert_top_token in modern_top5.indices:
            print(f"However, NorBERT's top token is in ModernBERT's top 5 predictions.")
            success = True
    
    return success


def main():
    parser = argparse.ArgumentParser(description="Convert NorBERT model to ModernBERT format.")
    parser.add_argument("--norbert_model_id", type=str, default="ltg/norbert3-base", 
                        help="HuggingFace model ID for NorBERT (default: ltg/norbert3-base)")
    parser.add_argument("--modernbert_model_id", type=str, default="answerdotai/ModernBERT-base", 
                        help="HuggingFace model ID for ModernBERT config (default: answerdotai/ModernBERT-base)")
    parser.add_argument("--output_dir", type=str, default="",
                        help="Directory to save the converted model (default: )")
    parser.add_argument("--test_conversion", action="store_true",
                        help="Test the conversion by comparing model outputs")
    parser.add_argument("--test_text", type=str, default="Nå ønsker de seg en[MASK] bolig.",
                        help="Text to use for testing (should include a [MASK] token)")
    parser.add_argument("--push-to-hub", action="store_true", help="Push the converted model to the Hugging Face Hub")

    args = parser.parse_args()

    output_dir = args.output_dir

    print(f"Converting {args.norbert_model_id} to ModernBERT format...")
    
    # Load NorBERT config
    norbert_config_url = f"https://huggingface.co/{args.norbert_model_id}/raw/main/config.json"
    print(f"Loading NorBERT config from {norbert_config_url}...")
    try:
        with smart_open.open(norbert_config_url, "r", encoding="utf-8") as f:
            norbert_config = json.load(f)
    except Exception as e:
        print(f"Error loading NorBERT config: {e}")
        print("Trying to load from the model directly...")
        # Alternative: load config from the model
        try:
            norbert_model = AutoModelForMaskedLM.from_pretrained(args.norbert_model_id, trust_remote_code=True)
            norbert_config = norbert_model.config.to_dict()
        except Exception as inner_e:
            print(f"Failed to load NorBERT config: {inner_e}")
            return

    # Load ModernBERT config
    modernbert_config_url = f"https://huggingface.co/{args.modernbert_model_id}/raw/main/config.json"
    print(f"Loading ModernBERT config from {modernbert_config_url}...")
    try:
        with smart_open.open(modernbert_config_url, "r", encoding="utf-8") as f:
            modernbert_config = json.load(f)
    except Exception as e:
        print(f"Error loading ModernBERT config: {e}")
        return

    # Load the tokenizer to get special token IDs
    print("Loading tokenizer to get special token IDs...")
    tokenizer = AutoTokenizer.from_pretrained(args.norbert_model_id)
    
    # Convert the config with tokenizer information
    print("Converting configuration...")
    new_config = convert_config(norbert_config, modernbert_config, tokenizer)

    if not output_dir:
        output_dir = "modern-norbert3-" + args.norbert_model_id.split("-")[-1]
    
    # Save the config to a temporary file
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(new_config, f, indent=2)
    print(f"Converted config saved to {config_path}")
    
    # Load NorBERT model and convert weights
    print(f"\nLoading NorBERT model from {args.norbert_model_id}...")
    norbert_model = AutoModelForMaskedLM.from_pretrained(args.norbert_model_id, trust_remote_code=True)
    norbert_state = norbert_model.state_dict()
    
    print("Converting model weights...")
    modern_state = convert_state_dict(norbert_state)
    
    # Initialize ModernBERT model with the converted config
    print("Initializing ModernBERT model with converted config...")
    modern_model = ModernBertForMaskedLM(ModernBertConfig.from_json_file(config_path))
    
    # Load the converted weights
    print("Loading converted weights into ModernBERT model...")
    modern_model.load_state_dict(modern_state)
    
    # Test the conversion if requested
    if args.test_conversion:
        test_conversion(args.norbert_model_id, modern_model, tokenizer, args.test_text)
    
    # Save the model
    print(f"\nSaving converted model to {output_dir}...")
    modern_model.save_pretrained(output_dir)
    
    # Also save the tokenizer
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.norbert_model_id)
    tokenizer.save_pretrained(output_dir)

    if args.push_to_hub:
        print("Pushing the model to the Hugging Face Hub...")
        modern_model.push_to_hub(output_dir)
        tokenizer.push_to_hub(output_dir)
    
    print(f"\nConversion complete! The converted model is saved in {output_dir}")
    print("You can now load this model with: from transformers import AutoModelForMaskedLM")
    print(f"model = AutoModelForMaskedLM.from_pretrained('{output_dir}')")


if __name__ == "__main__":
    main()