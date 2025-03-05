#!/usr/bin/env python
"""
A conversion script for mapping NorBERT (norbert3) weights to ModernBERT weights.

Usage:
    python convert_norbert_to_modernbert.py --norbert_path PATH_TO_NORBERT_WEIGHTS.pt \
                                             --modernbert_path PATH_TO_SAVE_MODERNBERT.pt
"""

import torch
import argparse
import transformers
from transformers import ModernBertConfig, ModernBertForMaskedLM

def convert_state_dict(norbert_state_dict):
    """
    Convert a NorBERT state dict into a ModernBERT state dict.
    
    The conversion does the following mappings:
    
    1. Embeddings:
       - NorBERT "embedding.word_embedding.weight" -> ModernBERT "model.embeddings.tok_embeddings.weight"
       - NorBERT "embedding.relative_layer_norm.weight" -> ModernBERT "model.embeddings.norm.weight"
         (NorBERT’s norm bias is dropped because ModernBERT’s state dict only contains a weight.)
       - NorBERT’s "embedding.relative_embedding" is not used by ModernBERT and is dropped.
    
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
             * (NorBERT’s out_proj bias is dropped.)
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
         * A head norm "head.norm.weight" is created (initialized to ones) since ModernBERT’s state dict contains it.
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
        # Get NorBERT’s fused Q/K weight and V weight:
        w_qk = norbert_state_dict[f"{norbert_prefix}.attention.in_proj_qk.weight"]  # shape: [2*H, H]
        w_v  = norbert_state_dict[f"{norbert_prefix}.attention.in_proj_v.weight"]    # shape: [H, H]
        # Concatenate along the first dimension to get shape [3*H, H] in the order [Q, K, V]
        w_qkv = torch.cat([w_qk, w_v], dim=0)
        new_state_dict[f"{modern_prefix}.attn.Wqkv.weight"] = w_qkv
        
        # --- Attention output projection ---
        new_state_dict[f"{modern_prefix}.attn.Wo.weight"] = norbert_state_dict[f"{norbert_prefix}.attention.out_proj.weight"]
        
        # --- Attention layer normalization ---
        if i > 0:
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

def main():
    model = transformers.AutoModelForMaskedLM.from_pretrained("ltg/norbert3-base", trust_remote_code=True)
    print(model)
    norbert_state = model.state_dict()
    # print(norbert_state)
    # exit()    
    print("Converting state dict ...")
    modern_state = convert_state_dict(norbert_state)
    # print(modern_state)

    modern_model = ModernBertForMaskedLM(ModernBertConfig.from_json_file("/home/markus/sandbox/output_manually_hard_coded.json"))
    print(modern_model)
    blank_state = modern_model.state_dict()
    print(blank_state["decoder.bias"])
    modern_model.load_state_dict(modern_state)
    print(blank_state["decoder.bias"])
    modern_model.save_pretrained("output_model")

    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    tokenizer = AutoTokenizer.from_pretrained("ltg/norbert3-base")
    model = AutoModelForMaskedLM.from_pretrained("ltg/norbert3-base", trust_remote_code=True)

    mask_id = tokenizer.convert_tokens_to_ids("[MASK]")
    input_text = tokenizer("Nå ønsker de seg en[MASK] bolig.", return_tensors="pt")
    output_p = model(**input_text)
    print(output_p)
    # output_text = torch.where(input_text.input_ids == mask_id, output_p.logits.argmax(-1), input_text.input_ids)

    # should output: '[CLS] Nå ønsker de seg en ny bolig.[SEP]'
    # print(tokenizer.decode(output_text[0].tolist()))

    mask_id = tokenizer.convert_tokens_to_ids("[MASK]")
    input_text = tokenizer("Nå ønsker de seg en[MASK] bolig.", return_tensors="pt")
    output_p = modern_model(**input_text)
    print(output_p)

    # should output: '[CLS] Nå ønsker de seg en ny bolig.[SEP]'
    # print(tokenizer.decode(output_text[0].tolist()))


if __name__ == "__main__":
    main()
