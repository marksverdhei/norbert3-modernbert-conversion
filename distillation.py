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

# Parse command line arguments
parser = argparse.ArgumentParser(description='Full Model Distillation from Norbert to ModernBERT')
parser.add_argument('--size', type=str, default='xs', choices=['xs', 'base', 'large'],
                    help='Model size (xs, base, large)')
parser.add_argument('--batch-size', type=int, default=16,
                    help='Training batch size')
parser.add_argument('--seq-len', type=int, default=128,
                    help='Sequence length for training')
parser.add_argument('--total-steps', type=int, default=60000,
                    help='Total number of training steps')
parser.add_argument('--warmup-steps', type=int, default=3000,
                    help='Number of warmup steps for learning rate scheduler')
parser.add_argument('--eval-every', type=int, default=500,
                    help='Evaluate every N steps')
parser.add_argument('--save-every', type=int, default=2000,
                    help='Save checkpoint every N steps')
parser.add_argument('--log-every', type=int, default=100,
                    help='Log metrics every N steps')
parser.add_argument('--embedding-noise', type=float, default=0.0,
                    help='Amount of noise to add to sampled embeddings (default: 0.0)')
parser.add_argument('--use-embeddings', action='store_true', default=True,
                    help='Sample from embedding layer instead of random vectors')
parser.add_argument('--output-dir', type=str, default='./modernbert_full_distillation',
                    help='Output directory for saved models')
parser.add_argument('--learning-rate', type=float, default=2e-5,
                    help='Learning rate for the optimizer')

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
# modernbert = AutoModelForMaskedLM.from_pretrained(f"marksverdhei/modern-norbert3-{args.size}")
modernbert = AutoModelForMaskedLM.from_pretrained(f"modernbert_progressive_vast/best/")
tokenizer = AutoTokenizer.from_pretrained(f"ltg/norbert3-{args.size}")

# Move models to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
norbert = norbert.to(device)
modernbert = modernbert.to(device)

# Set Norbert model to evaluation mode (we don't train it)
norbert.eval()

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

def generate_random_batch(batch_size, seq_len, hidden_size, use_embeddings=False, embedding_noise=0.0):
    """Generate random input vectors with shape [batch_size, seq_len, hidden_size]"""
    if use_embeddings:
        # Sample from the embedding layer instead of generating random vectors
        # Get vocabulary size and special token IDs
        vocab_size = modernbert.model.embeddings.tok_embeddings.num_embeddings
        cls_token_id = tokenizer.cls_token_id
        sep_token_id = tokenizer.sep_token_id
        mask_token_id = tokenizer.mask_token_id
        
        # Initialize token IDs - ensure we have CLS at start, SEP at end, and at least one MASK
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        # Set CLS and SEP tokens
        token_ids[:, 0] = cls_token_id
        token_ids[:, -1] = sep_token_id
        
        # Place at least one MASK token at a random position (not first or last)
        for i in range(batch_size):
            mask_pos = random.randint(1, seq_len - 2)  # Random position between CLS and SEP
            token_ids[i, mask_pos] = mask_token_id
        
        # Get embeddings for these tokens
        with torch.no_grad():
            # Use ModernBERT's embedding layer
            random_vectors = modernbert.model.embeddings.tok_embeddings(token_ids)
            
            # Apply layer norm and dropout if they are part of the embedding process
            random_vectors = modernbert.model.embeddings.norm(random_vectors)
            
            # Add noise if specified
            if embedding_noise > 0:
                noise = torch.randn_like(random_vectors) * embedding_noise
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

def get_norbert_outputs(input_vectors, attention_mask):
    """Get final outputs from the Norbert model"""
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
        
        return last_hidden_state, predictions

def get_modernbert_outputs(input_vectors, attention_mask):
    """Get final outputs from the ModernBERT model"""
    # We'll bypass the embedding layer and directly input to the encoder
    outputs = modernbert.model(
        inputs_embeds=input_vectors,
        attention_mask=attention_mask,
        output_hidden_states=False,
        return_dict=True
    )
    
    last_hidden_state = outputs.last_hidden_state
    predictions = modernbert.decoder(modernbert.head(last_hidden_state))
    
    return last_hidden_state, predictions

def evaluate(num_batches=10, batch_size=8, seq_len=128, use_embeddings=False, embedding_noise=0.0):
    """Evaluate models and return metrics"""
    modernbert.eval()
    
    total_hidden_mse = 0.0
    total_logits_mse = 0.0
    total_logits_kl = 0.0
    
    mse_loss = nn.MSELoss()
    kl_loss = nn.KLDivLoss(reduction='batchmean', log_target=False)
    
    with torch.no_grad():
        for _ in range(num_batches):
            input_vectors, attention_mask = generate_random_batch(
                batch_size=batch_size,
                seq_len=seq_len,
                hidden_size=modernbert.config.hidden_size,
                use_embeddings=use_embeddings,
                embedding_noise=embedding_noise
            )
            
            # Get outputs from both models
            norbert_hidden, norbert_logits = get_norbert_outputs(input_vectors, attention_mask)
            modernbert_hidden, modernbert_logits = get_modernbert_outputs(input_vectors, attention_mask)
            
            # Calculate hidden state MSE
            # hidden_mse = mse_loss(modernbert_hidden, norbert_hidden)
            # total_hidden_mse += hidden_mse.item()
            
            # Calculate logits MSE
            logits_mse = mse_loss(modernbert_logits, norbert_logits)
            total_logits_mse += logits_mse.item()
            
            # KL divergence for probability distributions
            norbert_probs = torch.nn.functional.softmax(norbert_logits, dim=-1)
            modernbert_log_probs = torch.nn.functional.log_softmax(modernbert_logits, dim=-1)
            logits_kl = kl_loss(modernbert_log_probs, norbert_probs)
            total_logits_kl += logits_kl.item()
    
    # Run masked token prediction evaluation
    norbert_prediction = evaluate_masked_prediction(norbert)
    modernbert_prediction = evaluate_masked_prediction(modernbert)
    
    modernbert.train()
    
    # Calculate average metrics
    avg_hidden_mse = total_hidden_mse / num_batches
    avg_logits_mse = total_logits_mse / num_batches
    avg_logits_kl = total_logits_kl / num_batches
    
    return {
        'hidden_mse': avg_hidden_mse,
        'logits_mse': avg_logits_mse,
        'logits_kl': avg_logits_kl,
        'norbert_prediction': norbert_prediction,
        'modernbert_prediction': modernbert_prediction
    }

def train_full_model_distillation():
    """Train using full model distillation - optimizing only on the final outputs"""
    # Define loss functions
    mse_loss = nn.MSELoss()
    kl_loss = nn.KLDivLoss(reduction='batchmean', log_target=False)
    
    # Configure optimizer - all parameters with the same learning rate
    optimizer = optim.AdamW(modernbert.parameters(), lr=args.learning_rate, weight_decay=0.01)
    
    # Create learning rate scheduler with warmup
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_steps)
    decay_scheduler = CosineAnnealingLR(optimizer, T_max=args.total_steps - args.warmup_steps, eta_min=1e-6)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[args.warmup_steps]
    )
    
    # Track best evaluation metrics
    best_eval_loss = float('inf')
    
    # Track start time
    start_time = time.time()
    
    # Training loop
    modernbert.train()
    progress_bar = tqdm(range(args.total_steps), desc="Training")
    
    for step in progress_bar:
        # Generate random batch
        input_vectors, attention_mask = generate_random_batch(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            hidden_size=modernbert.config.hidden_size,
            use_embeddings=args.use_embeddings,
            embedding_noise=args.embedding_noise
        )
        
        # Forward pass through both models
        norbert_hidden, norbert_logits = get_norbert_outputs(input_vectors, attention_mask)
        modernbert_hidden, modernbert_logits = get_modernbert_outputs(input_vectors, attention_mask)
        
        # Calculate hidden state MSE (for final layer only)
        hidden_mse = mse_loss(modernbert_hidden, norbert_hidden)
        
        # Calculate logits MSE
        logits_mse = mse_loss(modernbert_logits, norbert_logits)
        
        # KL divergence on probability distributions
        norbert_probs = torch.nn.functional.softmax(norbert_logits, dim=-1)
        modernbert_log_probs = torch.nn.functional.log_softmax(modernbert_logits, dim=-1)
        logits_kl = kl_loss(modernbert_log_probs, norbert_probs)
        
        # Combine losses with equal weighting
        loss = logits_mse + (0.5 * logits_kl)
        
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
            'logits': f"{logits_mse.item():.4f}",
            'kl': f"{logits_kl.item():.4f}"
        })
        
        # Log metrics
        if step % args.log_every == 0:
            writer.add_scalar('training/loss', loss.item(), step)
            writer.add_scalar('training/hidden_mse', hidden_mse.item(), step)
            writer.add_scalar('training/logits_mse', logits_mse.item(), step)
            writer.add_scalar('training/logits_kl', logits_kl.item(), step)
            writer.add_scalar('training/lr', scheduler.get_last_lr()[0], step)
        
        # Evaluate
        if step % args.eval_every == 0 or step == args.total_steps - 1:
            eval_metrics = evaluate(
                num_batches=10,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                use_embeddings=args.use_embeddings,
                embedding_noise=args.embedding_noise
            )
            
            # Log evaluation metrics
            writer.add_scalar('validation/hidden_mse', eval_metrics['hidden_mse'], step)
            writer.add_scalar('validation/logits_mse', eval_metrics['logits_mse'], step)
            writer.add_scalar('validation/logits_kl', eval_metrics['logits_kl'], step)
            
            # Log masked prediction metrics
            modernbert_pred = eval_metrics['modernbert_prediction']
            writer.add_scalar('masked_prediction/target_rank', modernbert_pred['target_rank'], step)
            writer.add_scalar('masked_prediction/target_prob', modernbert_pred['target_prob'], step)
            
            # Print detailed evaluation metrics
            if step % (args.eval_every * 5) == 0 or step == args.total_steps - 1:
                norbert_pred = eval_metrics['norbert_prediction']
                modernbert_pred = eval_metrics['modernbert_prediction']
                
                tqdm.write(f"\nStep {step}: Eval "
                          f"logits_mse={eval_metrics['logits_mse']:.6f}, "
                          f"logits_kl={eval_metrics['logits_kl']:.6f}")
                
                # Print masked prediction results
                tqdm.write("\nMasked token prediction:")
                tqdm.write(f"  ModernBERT output: '{modernbert_pred['predicted_text']}'")
                tqdm.write(f"  Target token 'ny' rank: {modernbert_pred['target_rank']}, "
                          f"probability: {modernbert_pred['target_prob']:.4f}")
            
            # Calculate combined evaluation loss
            eval_loss = eval_metrics['logits_mse'] + eval_metrics['logits_kl']
            
            # Save if best model
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                save_path = save_model(modernbert, f"{args.output_dir}/best")
                if save_path:
                    tqdm.write(f"\nSaved new best model with eval_loss={eval_loss:.6f}")
        
        # Save checkpoint
        if step > 0 and step % args.save_every == 0:
            save_model(modernbert, args.output_dir, f"/step_{step}")
    
    # Final full evaluation
    final_metrics = evaluate(
        num_batches=50,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        use_embeddings=args.use_embeddings,
        embedding_noise=args.embedding_noise
    )
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Print final results
    tqdm.write("\n\n=== Full Model Distillation Complete ===")
    tqdm.write(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
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
    print(f"Starting full model distillation from Norbert3-{args.size} to ModernBERT-{args.size}")
    print(f"Using device: {device}")
    print(f"Training with {args.total_steps} total steps")
    
    if args.use_embeddings:
        print("Using embeddings from the model for input generation")
    else:
        print("Using random vectors for input generation")
    
    train_full_model_distillation()
    
    print("\nFull model distillation completed successfully!")

# Example command to run this script:
# python full_model_distillation.py --size xs --batch-size 16 --seq-len 128 --total-steps 60000 --use-embeddings