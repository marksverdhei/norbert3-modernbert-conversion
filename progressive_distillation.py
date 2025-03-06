import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm.auto import tqdm
import random
import numpy as np
import argparse
import os
import time
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Define constants
N_BLOCKS = 12  # Number of encoder blocks

# Parse command line arguments
parser = argparse.ArgumentParser(description='Progressive Distillation from Norbert to ModernBERT')
parser.add_argument('--size', type=str, default='xs', choices=['xs', 'base', 'large'],
                    help='Model size (xs, base, large)')
parser.add_argument('--batch-size', type=int, default=16,
                    help='Training batch size')
parser.add_argument('--seq-len', type=int, default=128,
                    help='Sequence length for training')
parser.add_argument('--steps-per-phase', type=int, default=5000,
                    help='Number of training steps per distillation phase')
parser.add_argument('--warmup-steps', type=int, default=500,
                    help='Number of warmup steps for learning rate scheduler')
parser.add_argument('--eval-every', type=int, default=500,
                    help='Evaluate every N steps')
parser.add_argument('--save-every', type=int, default=2000,
                    help='Save checkpoint every N steps')
parser.add_argument('--log-every', type=int, default=100,
                    help='Log metrics every N steps')
parser.add_argument('--use-embeddings', action='store_true', default=True,
                    help='Sample from embedding layer instead of random vectors')
parser.add_argument('--output-dir', type=str, default='./modernbert_progressive',
                    help='Output directory for saved models')

args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(f"{args.output_dir}/logs", exist_ok=True)

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir=f"{args.output_dir}/logs")

# Load models and tokenizer
norbert = AutoModelForMaskedLM.from_pretrained(f"ltg/norbert3-{args.size}", trust_remote_code=True)
modernbert = AutoModelForMaskedLM.from_pretrained(f"marksverdhei/modern-norbert3-{args.size}")
tokenizer = AutoTokenizer.from_pretrained(f"ltg/norbert3-{args.size}")

# Move models to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
norbert = norbert.to(device)
modernbert = modernbert.to(device)

# Set Norbert model to evaluation mode (we don't train it)
norbert.eval()

# Freeze embedding layer in ModernBERT since we're skipping it
modernbert.model.embeddings.requires_grad_(False)

def save_model(model, base_path, suffix=""):
    """
    Save the model with error handling.
    If saving fails, try a different path with a counter appended.
    """
    counter = 0
    while True:
        try:
            path = f"{base_path}{suffix}"
            if counter > 0:
                path = f"{path}_{counter}"
            
            model.save_pretrained(path)
            tqdm.write(f"Model saved to {path}")
            return path
        except Exception as e:
            counter += 1
            tqdm.write(f"Error saving model to {path}: {str(e)}")
            if counter >= 5:  # Try a maximum of 5 alternative paths
                tqdm.write("Failed to save model after multiple attempts")
                return None
            tqdm.write(f"Trying alternative path...")
            continue

def evaluate_masked_prediction(model, example_text="Nå ønsker de seg en[MASK] bolig.", target_token_id=565):
    """Evaluate a model on a specific masked prediction task"""
    model.eval()
    mask_id = tokenizer.convert_tokens_to_ids("[MASK]")
    inputs = tokenizer(example_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Get predictions for the masked token
    mask_positions = (inputs.input_ids == mask_id).nonzero(as_tuple=True)[1]
    mask_position = mask_positions[0].item()
    
    # Get top 5 predictions and their probabilities
    mask_logits = outputs.logits[0, mask_position]
    probs = torch.nn.functional.softmax(mask_logits, dim=-1)
    
    top_5_probs, top_5_indices = torch.topk(probs, 5)
    top_5_tokens = [tokenizer.convert_ids_to_tokens(idx.item()) for idx in top_5_indices]
    
    # Get the rank and probability of the target token
    target_prob = probs[target_token_id].item()
    target_rank = (probs > target_prob).sum().item() + 1
    
    # Decode the sequence with the top prediction
    predicted_ids = torch.where(
        inputs.input_ids == mask_id, 
        outputs.logits.argmax(-1), 
        inputs.input_ids
    )
    predicted_text = tokenizer.decode(predicted_ids[0].tolist())
    
    results = {
        'top_5_tokens': top_5_tokens,
        'top_5_probs': top_5_probs.tolist(),
        'target_rank': target_rank,
        'target_prob': target_prob,
        'predicted_text': predicted_text
    }
    
    return results

def generate_random_batch(batch_size, seq_len, hidden_size, use_embeddings=False):
    """Generate random input vectors with shape [batch_size, seq_len, hidden_size]"""
    if use_embeddings:
        # Sample from the embedding layer instead of generating random vectors
        # Get vocabulary size
        vocab_size = modernbert.model.embeddings.tok_embeddings.num_embeddings
        
        # Generate random token IDs
        random_token_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        # Get embeddings for these tokens
        with torch.no_grad():
            # Use ModernBERT's embedding layer
            random_vectors = modernbert.model.embeddings.tok_embeddings(random_token_ids)
            
            # Apply layer norm and dropout if they are part of the embedding process
            random_vectors = modernbert.model.embeddings.norm(random_vectors)
            
            # Add a bit of noise for stability and to avoid perfect matching
            noise = torch.randn_like(random_vectors) * 0.01
            random_vectors = random_vectors + noise
    else:
        # Generate with small values to ensure stability
        random_vectors = torch.randn(batch_size, seq_len, hidden_size, device=device) * 0.1
    
    # Create random attention mask (1 for tokens to attend to, 0 for padding)
    # Ensure at least half the sequence is used
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    for i in range(batch_size):
        pad_len = random.randint(0, seq_len // 2)
        if pad_len > 0:
            attention_mask[i, -pad_len:] = 0
            
    return random_vectors, attention_mask

def get_norbert_outputs_with_intermediates(input_vectors, attention_mask):
    """Get outputs and intermediate activations from the Norbert model"""
    with torch.no_grad():
        # For Norbert, we need to invert the attention mask (1 for padding, 0 for tokens)
        norbert_mask = ~attention_mask.bool()
        norbert_mask = norbert_mask.unsqueeze(1).unsqueeze(2)
        
        # Get the relative embeddings from the Norbert model
        relative_embedding = norbert.embedding.relative_embedding
        relative_embedding = norbert.embedding.relative_layer_norm(relative_embedding)
        
        # Process through transformer layers - skipping the embedding layer
        hidden_states, attention_probs = norbert.transformer(
            input_vectors, norbert_mask, relative_embedding
        )
        
        # Get the last hidden state and run through the MLM head
        last_hidden_state = hidden_states[-1]
        predictions = norbert.classifier(last_hidden_state)
        
        return hidden_states, predictions

def get_modernbert_outputs_with_intermediates(input_vectors, attention_mask):
    """Get outputs and intermediate activations from the ModernBERT model"""
    # We'll bypass the embedding layer and directly input to the encoder
    outputs = modernbert.model(
        inputs_embeds=input_vectors,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True
    )
    
    last_hidden_state = outputs.last_hidden_state
    hidden_states = outputs.hidden_states
    predictions = modernbert.decoder(modernbert.head(last_hidden_state))
    
    return hidden_states, predictions

def evaluate(num_batches=10, batch_size=8, seq_len=128, use_embeddings=False, active_layers=None):
    """Evaluate models and return metrics"""
    modernbert.eval()
    
    total_hidden_mse_by_layer = {i: 0.0 for i in range(N_BLOCKS + 1)}  # +1 for final output
    total_logits_mse = 0.0
    total_logits_kl = 0.0
    
    mse_loss = nn.MSELoss()
    kl_loss = nn.KLDivLoss(reduction='batchmean', log_target=False)
    
    with torch.no_grad():
        for i in range(num_batches):
            input_vectors, attention_mask = generate_random_batch(
                batch_size=batch_size,
                seq_len=seq_len,
                hidden_size=modernbert.config.hidden_size,
                use_embeddings=use_embeddings
            )
            
            # Get outputs from both models with intermediate activations
            norbert_hidden_states, norbert_logits = get_norbert_outputs_with_intermediates(input_vectors, attention_mask)
            modernbert_hidden_states, modernbert_logits = get_modernbert_outputs_with_intermediates(input_vectors, attention_mask)
            
            # Calculate MSE for each layer
            for layer_idx in range(len(norbert_hidden_states)):
                if active_layers is None or layer_idx in active_layers:
                    layer_mse = mse_loss(modernbert_hidden_states[layer_idx], norbert_hidden_states[layer_idx])
                    total_hidden_mse_by_layer[layer_idx] += layer_mse.item()
            
            # Calculate logits MSE and KL divergence
            logits_mse = mse_loss(modernbert_logits, norbert_logits)
            
            # KL divergence for probability distributions
            norbert_probs = torch.nn.functional.softmax(norbert_logits, dim=-1)
            modernbert_log_probs = torch.nn.functional.log_softmax(modernbert_logits, dim=-1)
            logits_kl = kl_loss(modernbert_log_probs, norbert_probs)
            
            total_logits_mse += logits_mse.item()
            total_logits_kl += logits_kl.item()
    
    # Run masked token prediction evaluation
    norbert_prediction = evaluate_masked_prediction(norbert)
    modernbert_prediction = evaluate_masked_prediction(modernbert)
    
    modernbert.train()
    
    # Calculate average losses
    avg_hidden_mse_by_layer = {
        layer_idx: total / num_batches for layer_idx, total in total_hidden_mse_by_layer.items()
    }
    avg_logits_mse = total_logits_mse / num_batches
    avg_logits_kl = total_logits_kl / num_batches
    
    # Calculate overall hidden MSE based on active layers
    if active_layers is None:
        active_layers = list(range(N_BLOCKS + 1))
    avg_hidden_mse = sum(avg_hidden_mse_by_layer[i] for i in active_layers) / len(active_layers)
    
    return {
        'hidden_mse': avg_hidden_mse,
        'hidden_mse_by_layer': avg_hidden_mse_by_layer,
        'logits_mse': avg_logits_mse,
        'logits_kl': avg_logits_kl,
        'norbert_prediction': norbert_prediction,
        'modernbert_prediction': modernbert_prediction
    }

def train_progressive_distillation():
    """Train using progressive distillation - gradually include deeper layers"""
    # Define loss functions
    mse_loss = nn.MSELoss()
    kl_loss = nn.KLDivLoss(reduction='batchmean', log_target=False)
    
    # Number of phases - we'll have N_BLOCKS phases (one per layer activation)
    num_phases = N_BLOCKS
    steps_per_phase = args.steps_per_phase
    
    # Track best evaluation metrics
    best_eval_loss = float('inf')
    best_phase = -1
    
    # Track start time
    start_time = time.time()
    
    # Progressively activate layers
    for phase in range(num_phases):
        tqdm.write(f"\n\n=== Starting Phase {phase+1}/{num_phases} - Aligning Layers 0 to {phase} ===\n")
        active_layers = list(range(phase + 1))  # Layers 0 to phase
        
        # Configure optimizer with layer-wise learning rates for this phase
        optimizer_grouped_parameters = []
        
        # Only optimize the active layers and the layers immediately after
        active_layers_for_optim = active_layers + [phase + 1] if phase + 1 < N_BLOCKS else active_layers
        
        # For phase 0, also train the head since we need to get some signal from the output
        if phase == 0:
            # MLP Head gets higher learning rate
            optimizer_grouped_parameters.append({
                "params": [p for n, p in modernbert.named_parameters() if "head" in n and p.requires_grad],
                "lr": 5e-5,
                "weight_decay": 0.01
            })
        
        # Always train output projection for final prediction
        optimizer_grouped_parameters.append({
            "params": [p for n, p in modernbert.named_parameters() if "decoder" in n and p.requires_grad],
            "lr": 1e-5,
            "weight_decay": 0.01
        })
        
        # Encoder layers get learning rates based on phase
        for i in active_layers_for_optim:
            layer_params = [p for n, p in modernbert.named_parameters() 
                           if f"layers.{i}." in n and p.requires_grad]
            
            # Calculate learning rate - current phase layer gets highest rate
            if i == phase:
                layer_lr = 1e-4
            elif i < phase:
                # Previously activated layers get lower rates, decreasing with depth
                layer_lr = 5e-5 * (0.8 ** (phase - i))
            else:
                # Future layers that we're pre-conditioning get much lower rates
                layer_lr = 1e-5
            
            optimizer_grouped_parameters.append({
                "params": layer_params,
                "lr": layer_lr,
                "weight_decay": 0.01
            })
        
        # Create optimizer
        optimizer = optim.AdamW(optimizer_grouped_parameters)
        
        # Create learning rate scheduler with warmup
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_steps)
        decay_scheduler = CosineAnnealingLR(optimizer, T_max=steps_per_phase - args.warmup_steps, eta_min=1e-6)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[args.warmup_steps]
        )
        
        # Print active layers and learning rates
        tqdm.write("Active layers for this phase:")
        for i in active_layers:
            tqdm.write(f"  - Layer {i}")
            
        tqdm.write("\nLearning rates:")
        for i, group in enumerate(optimizer.param_groups):
            # Get a sample parameter from the group to identify it
            sample_param = None
            sample_name = "unknown"
            for name, param in modernbert.named_parameters():
                if any(p is param for p in group['params']):
                    sample_param = param
                    sample_name = name
                    break
                    
            if sample_param is not None:
                tqdm.write(f"  - Group {i}: LR={group['lr']:.6f} ({sample_name}...)")
        
        # Training loop for this phase
        modernbert.train()
        progress_bar = tqdm(range(steps_per_phase), desc=f"Phase {phase+1}")
        
        for step in progress_bar:
            global_step = phase * steps_per_phase + step
            
            # Generate random batch
            input_vectors, attention_mask = generate_random_batch(
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                hidden_size=modernbert.config.hidden_size,
                use_embeddings=args.use_embeddings
            )
            
            # Forward pass through both models and get intermediate activations
            norbert_hidden_states, norbert_logits = get_norbert_outputs_with_intermediates(input_vectors, attention_mask)
            modernbert_hidden_states, modernbert_logits = get_modernbert_outputs_with_intermediates(input_vectors, attention_mask)
            
            # Calculate losses for active layers only
            hidden_losses = []
            for i in active_layers:
                layer_loss = mse_loss(modernbert_hidden_states[i], norbert_hidden_states[i])
                hidden_losses.append(layer_loss)
            
            # Average the hidden state losses
            hidden_loss = sum(hidden_losses) / len(hidden_losses) if hidden_losses else torch.tensor(0.0).to(device)
            
            # Calculate logits loss (only important for later phases)
            logits_mse = mse_loss(modernbert_logits, norbert_logits)
            
            # KL divergence on probability distributions
            norbert_probs = torch.nn.functional.softmax(norbert_logits, dim=-1) 
            modernbert_log_probs = torch.nn.functional.log_softmax(modernbert_logits, dim=-1)
            logits_kl = kl_loss(modernbert_log_probs, norbert_probs)
            
            # Combine losses with phase-appropriate weighting
            # Early phases focus more on hidden states, later phases more on final output
            hidden_weight = 1.0 - (phase / (num_phases * 2))  # Gradually decreases from 1.0 to 0.5
            logits_weight = 0.5 + (phase / (num_phases * 2))  # Gradually increases from 0.5 to 1.0
            
            loss = (hidden_weight * hidden_loss) + (logits_weight * logits_mse) + (0.5 * logits_kl)
            
            # Backward pass and optimizer step
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(modernbert.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'hidden': f"{hidden_loss.item():.4f}",
                'logits': f"{logits_mse.item():.4f}",
                'kl': f"{logits_kl.item():.4f}"
            })
            
            # Log metrics
            if step % args.log_every == 0:
                writer.add_scalar(f'training/phase_{phase+1}/loss', loss.item(), global_step)
                writer.add_scalar(f'training/phase_{phase+1}/hidden_loss', hidden_loss.item(), global_step)
                writer.add_scalar(f'training/phase_{phase+1}/logits_mse', logits_mse.item(), global_step)
                writer.add_scalar(f'training/phase_{phase+1}/logits_kl', logits_kl.item(), global_step)
                writer.add_scalar(f'training/phase_{phase+1}/lr', scheduler.get_last_lr()[0], global_step)
                
                # Log individual layer losses
                for i, layer_loss in enumerate(hidden_losses):
                    writer.add_scalar(f'training/phase_{phase+1}/layer_{i}_loss', layer_loss.item(), global_step)
            
            # Evaluate
            if step % args.eval_every == 0 or step == steps_per_phase - 1:
                eval_metrics = evaluate(
                    num_batches=10,
                    batch_size=args.batch_size,
                    seq_len=args.seq_len,
                    use_embeddings=args.use_embeddings,
                    active_layers=active_layers
                )
                
                # Log evaluation metrics
                writer.add_scalar(f'validation/phase_{phase+1}/hidden_mse', eval_metrics['hidden_mse'], global_step)
                writer.add_scalar(f'validation/phase_{phase+1}/logits_mse', eval_metrics['logits_mse'], global_step)
                writer.add_scalar(f'validation/phase_{phase+1}/logits_kl', eval_metrics['logits_kl'], global_step)
                
                # Log layer-wise metrics
                for layer_idx, mse in eval_metrics['hidden_mse_by_layer'].items():
                    if layer_idx <= phase:  # Only log active layers
                        writer.add_scalar(f'validation/phase_{phase+1}/layer_{layer_idx}_mse', mse, global_step)
                
                # Log masked prediction metrics
                modernbert_pred = eval_metrics['modernbert_prediction']
                writer.add_scalar(f'masked_prediction/phase_{phase+1}/target_rank', 
                                  modernbert_pred['target_rank'], global_step)
                writer.add_scalar(f'masked_prediction/phase_{phase+1}/target_prob', 
                                  modernbert_pred['target_prob'], global_step)
                
                # Print detailed evaluation metrics
                if step % (args.eval_every * 5) == 0 or step == steps_per_phase - 1:
                    norbert_pred = eval_metrics['norbert_prediction']
                    modernbert_pred = eval_metrics['modernbert_prediction']
                    
                    tqdm.write(f"\nPhase {phase+1} - Step {step}: Eval hidden_mse={eval_metrics['hidden_mse']:.6f}, "
                              f"logits_mse={eval_metrics['logits_mse']:.6f}, "
                              f"logits_kl={eval_metrics['logits_kl']:.6f}")
                    
                    # Print layer-wise MSE
                    tqdm.write("\nLayer-wise MSE:")
                    for layer_idx in range(phase + 1):
                        tqdm.write(f"  - Layer {layer_idx}: {eval_metrics['hidden_mse_by_layer'][layer_idx]:.6f}")
                    
                    # Print masked prediction results
                    tqdm.write("\nMasked token prediction:")
                    tqdm.write(f"  ModernBERT output: '{modernbert_pred['predicted_text']}'")
                    tqdm.write(f"  Target token 'ny' rank: {modernbert_pred['target_rank']}, "
                              f"probability: {modernbert_pred['target_prob']:.4f}")
                
                # Calculate combined evaluation loss
                eval_loss = eval_metrics['hidden_mse'] + eval_metrics['logits_mse'] + eval_metrics['logits_kl']
                
                # Save if best model for this phase
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    best_phase = phase
                    save_path = save_model(modernbert, f"{args.output_dir}/best")
                    if save_path:
                        tqdm.write(f"\nSaved new best model with eval_loss={eval_loss:.6f}")
            
            # Save checkpoint
            if step > 0 and step % args.save_every == 0:
                save_model(modernbert, args.output_dir, f"/phase_{phase+1}_step_{step}")
        
        # End of phase - save model
        save_model(modernbert, args.output_dir, f"/phase_{phase+1}_final")
        
        # Full evaluation at the end of the phase with all layers
        tqdm.write("\n=== End of Phase Evaluation ===")
        end_phase_metrics = evaluate(
            num_batches=20,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            use_embeddings=args.use_embeddings,
            active_layers=None  # Evaluate all layers
        )
        
        # Display comprehensive metrics
        tqdm.write(f"\nComprehensive evaluation after Phase {phase+1}:")
        tqdm.write(f"Overall hidden_mse: {end_phase_metrics['hidden_mse']:.6f}")
        tqdm.write(f"Logits MSE: {end_phase_metrics['logits_mse']:.6f}")
        tqdm.write(f"Logits KL: {end_phase_metrics['logits_kl']:.6f}")
        
        tqdm.write("\nLayer-wise hidden state MSE:")
        for layer_idx in range(N_BLOCKS + 1):
            status = "✓" if layer_idx <= phase else "×"
            tqdm.write(f"  Layer {layer_idx}: {end_phase_metrics['hidden_mse_by_layer'][layer_idx]:.6f} {status}")
    
    # Final full evaluation
    final_metrics = evaluate(
        num_batches=50,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        use_embeddings=args.use_embeddings
    )
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Print final results
    tqdm.write("\n\n=== Progressive Distillation Complete ===")
    tqdm.write(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    tqdm.write(f"Best model from phase: {best_phase + 1}")
    tqdm.write(f"Best evaluation loss: {best_eval_loss:.6f}")
    
    tqdm.write("\nFinal evaluation metrics:")
    tqdm.write(f"  Hidden MSE: {final_metrics['hidden_mse']:.6f}")
    tqdm.write(f"  Logits MSE: {final_metrics['logits_mse']:.6f}")
    tqdm.write(f"  Logits KL: {final_metrics['logits_kl']:.6f}")
    
    # Display final masked prediction results
    norbert_pred = final_metrics['norbert_prediction']
    modernbert_pred = final_metrics['modernbert_prediction']
    
    tqdm.write("\nFinal masked token prediction comparison:")
    tqdm.write(f"Input text: 'Nå ønsker de seg en[MASK] bolig.'")
    
    # Norbert predictions
    tqdm.write("\nNorbert predictions:")
    tqdm.write(f"Output: '{norbert_pred['predicted_text']}'")
    for i, (token, prob) in enumerate(zip(norbert_pred['top_5_tokens'], norbert_pred['top_5_probs'])):
        tqdm.write(f"  {i+1}. '{token}' ({prob:.4f})")
    tqdm.write(f"Target token 'ny' rank: {norbert_pred['target_rank']}, probability: {norbert_pred['target_prob']:.4f}")
    
    # ModernBERT predictions
    tqdm.write("\nModernBERT predictions:")
    tqdm.write(f"Output: '{modernbert_pred['predicted_text']}'")
    for i, (token, prob) in enumerate(zip(modernbert_pred['top_5_tokens'], modernbert_pred['top_5_probs'])):
        tqdm.write(f"  {i+1}. '{token}' ({prob:.4f})")
    tqdm.write(f"Target token 'ny' rank: {modernbert_pred['target_rank']}, probability: {modernbert_pred['target_prob']:.4f}")
    
    # Save final model
    save_model(modernbert, f"{args.output_dir}/final")
    
    # Close tensorboard writer
    writer.close()

if __name__ == "__main__":
    print(f"Starting progressive distillation from Norbert3-{args.size} to ModernBERT-{args.size}")
    print(f"Using device: {device}")
    print(f"Training with {args.steps_per_phase} steps per phase × {N_BLOCKS} phases")
    
    if args.use_embeddings:
        print("Using embeddings from the model for input generation")
    else:
        print("Using random vectors for input generation")
    
    train_progressive_distillation()
    
    print("\nProgressive distillation completed successfully!")

# Example command to run this script:
# python progressive_distillation.py --size xs --batch-size 16 --seq-len 128 --steps-per-phase 5000 --use-embeddings