# norbert3-modernbert-conversion
Scripts for converting weights and tuning Transformer blocks

This Repository contains the code for converting the weights of Norbert 3 models to ModernBERT. 
Note that a 1-to-1 mapping does not work directly because of architectural differences, 
so the model can be tuned/distilled on noise to approximate the original model.

Training script is in progress.