# table5_paper_accurate.py - Accurate implementation following the paper's attention modes
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaForMaskedLM, get_linear_schedule_with_warmup
from torch.optim import AdamW
import logging
import random
from tqdm import tqdm
import argparse
import sys
import json

# Add Longformer to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'longformer'))
from longformer import LongformerForMaskedLM, LongformerConfig

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class MLMDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_length):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.examples = []
        
        logger.info(f"Loading data from {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        # Split into documents
        docs = text.split('\n\n')
        logger.info(f"Found {len(docs)} documents")
        
        # Process each document
        for doc in tqdm(docs, desc="Processing documents"):
            if len(doc.strip()) > 0:
                tokens = tokenizer.tokenize(doc)
                token_ids = tokenizer.convert_tokens_to_ids(tokens)
                
                # Split into chunks of max_seq_length
                for i in range(0, len(token_ids), seq_length):
                    chunk = token_ids[i:i + seq_length]
                    if len(chunk) >= 16:  # Minimum length to avoid issues
                        # Pad if necessary
                        if len(chunk) < seq_length:
                            chunk = chunk + [tokenizer.pad_token_id] * (seq_length - len(chunk))
                        self.examples.append(chunk)
        
        logger.info(f"Created {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        tokens = torch.tensor(tokens, dtype=torch.long)
        
        # Create masked version for MLM
        masked_tokens, labels = self.mask_tokens(tokens)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (tokens != self.tokenizer.pad_token_id).long()
        
        return {
            "input_ids": masked_tokens,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    def mask_tokens(self, inputs):
        """Prepare masked tokens for MLM training with guaranteed masking."""
        labels = inputs.clone()
        
        # Create a mask for special tokens (these should not be masked)
        special_token_ids = {
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.unk_token_id,
            self.tokenizer.mask_token_id
        }
        
        # Find positions that can be masked (not special tokens)
        maskable_positions = torch.tensor([
            token_id not in special_token_ids for token_id in inputs.tolist()
        ], dtype=torch.bool)
        
        # Ensure we have some maskable positions
        if not maskable_positions.any():
            # If no maskable positions, mask everything except pad tokens
            maskable_positions = (inputs != self.tokenizer.pad_token_id)
        
        # Create probability matrix for masking
        probability_matrix = torch.zeros_like(inputs, dtype=torch.float)
        probability_matrix[maskable_positions] = 0.15
        
        # Generate masked indices
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # Ensure at least one token is masked (unless all are special tokens)
        if not masked_indices.any() and maskable_positions.any():
            # Randomly select one maskable position
            maskable_indices = maskable_positions.nonzero().flatten()
            random_idx = torch.randint(0, len(maskable_indices), (1,))
            masked_indices[maskable_indices[random_idx]] = True
        
        # Set labels: -100 for non-masked tokens (ignored in loss calculation)
        labels[~masked_indices] = -100
        
        # Replace masked tokens:
        # 80% of the time, replace with [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id
        
        # 10% of the time, replace with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        
        # 10% of the time, keep original token
        
        return inputs, labels

def calculate_bpc(model, dataloader, device, max_batches=None):
    """Calculate bits per character (BPC) metric on the dataset."""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    nan_batches = 0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Calculating BPC")):
            if max_batches and i >= max_batches:
                break
                
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Check if this batch has any tokens to predict
            num_masked_tokens = (batch["labels"] != -100).sum().item()
            
            if num_masked_tokens == 0:
                continue
            
            try:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                
                loss = outputs.loss.item()
                
                if torch.isnan(torch.tensor(loss)) or torch.isinf(torch.tensor(loss)):
                    nan_batches += 1
                    continue
                else:
                    total_loss += loss
                    total_batches += 1
                    
            except Exception as e:
                logger.warning(f"Error processing batch {i}: {str(e)}")
                continue
    
    if total_batches == 0:
        logger.error("No valid batches processed!")
        return float('nan')
    
    # Calculate average loss and convert to BPC
    avg_loss = total_loss / total_batches
    bpc = avg_loss / np.log(2)
    
    if nan_batches > 0:
        logger.warning(f"Skipped {nan_batches} batches due to NaN/inf loss")
    
    model.train()
    return bpc

def create_longformer_config(roberta_config, max_pos, attention_window=512):
    """Create a Longformer config based on a RoBERTa config with proper sliding window attention."""
    
    # Use sliding window attention as in the paper
    # Try different attention modes in order of preference
    attention_modes = ["sliding_chunks", "sliding_chunks_no_overlap", "n2"]
    
    for attention_mode in attention_modes:
        try:
            config = LongformerConfig(
                attention_window=[attention_window] * roberta_config.num_hidden_layers,
                attention_dilation=[1] * roberta_config.num_hidden_layers,
                autoregressive=False,
                attention_mode=attention_mode,
                vocab_size=roberta_config.vocab_size,
                hidden_size=roberta_config.hidden_size,
                num_hidden_layers=roberta_config.num_hidden_layers,
                num_attention_heads=roberta_config.num_attention_heads,
                intermediate_size=roberta_config.intermediate_size,
                hidden_act=roberta_config.hidden_act,
                hidden_dropout_prob=roberta_config.hidden_dropout_prob,
                attention_probs_dropout_prob=roberta_config.attention_probs_dropout_prob,
                max_position_embeddings=max_pos,
                type_vocab_size=roberta_config.type_vocab_size,
                initializer_range=roberta_config.initializer_range,
                layer_norm_eps=roberta_config.layer_norm_eps,
            )
            
            logger.info(f"Using attention mode: {attention_mode}")
            if attention_mode == "n2":
                logger.warning("Using n2 attention mode - this is not the sliding window attention from the paper!")
            
            return config
            
        except Exception as e:
            logger.warning(f"Failed to create config with attention mode {attention_mode}: {str(e)}")
            continue
    
    raise RuntimeError("Failed to create Longformer config with any attention mode")

def extend_position_embeddings(model, max_pos, copy=True):
    """Extend position embeddings to max_pos using the method from the paper."""
    embeddings = model.roberta.embeddings.position_embeddings
    old_max_pos = embeddings.weight.size(0)
    embedding_dim = embeddings.weight.size(1)
    
    if max_pos <= old_max_pos:
        logger.info(f"Position embeddings already support {max_pos} positions")
        return model
    
    # Create new position embeddings
    new_embeddings = torch.nn.Embedding(max_pos, embedding_dim)
    new_embeddings.to(embeddings.weight.device)
    
    # Initialize weights according to the paper's method
    with torch.no_grad():
        if copy:
            logger.info(f"Copying position embeddings from {old_max_pos} to {max_pos} (as in paper)")
            
            # As described in the paper: "we initialize them by copying the 512 position 
            # embeddings from RoBERTa multiple times"
            
            # Copy the original embeddings first
            new_embeddings.weight.data[:old_max_pos] = embeddings.weight.data
            
            # For the rest, copy the embeddings multiple times
            # Skip the first 2 positions which are special (CLS, SEP)
            copy_start = 2
            copy_length = old_max_pos - copy_start
            
            remaining_positions = max_pos - old_max_pos
            full_copies = remaining_positions // copy_length
            remainder = remaining_positions % copy_length
            
            # Copy full chunks
            for i in range(full_copies):
                start_idx = old_max_pos + i * copy_length
                end_idx = start_idx + copy_length
                new_embeddings.weight.data[start_idx:end_idx] = embeddings.weight.data[copy_start:old_max_pos]
            
            # Copy the remainder
            if remainder > 0:
                start_idx = old_max_pos + full_copies * copy_length
                end_idx = start_idx + remainder
                new_embeddings.weight.data[start_idx:end_idx] = embeddings.weight.data[copy_start:copy_start + remainder]
                
        else:
            logger.info(f"Randomly initializing position embeddings from {old_max_pos} to {max_pos}")
            # Keep original embeddings, randomly initialize the rest
            new_embeddings.weight.data[:old_max_pos] = embeddings.weight.data
            # The rest will be randomly initialized by default
    
    # Replace the embeddings
    model.roberta.embeddings.position_embeddings = new_embeddings
    model.roberta.embeddings.register_buffer(
        "position_ids", torch.arange(max_pos).expand((1, -1))
    )
    
    # Update max position in config
    model.config.max_position_embeddings = max_pos
    
    return model

def test_model_compatibility(model, dataloader, device):
    """Test if the model works with a single batch to check compatibility."""
    model.eval()
    try:
        # Get first batch
        batch = next(iter(dataloader))
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss.item()
            
        if torch.isnan(torch.tensor(loss)) or torch.isinf(torch.tensor(loss)):
            return False, f"NaN/inf loss: {loss}"
        
        logger.info(f"Model compatibility test passed. Test loss: {loss:.4f}")
        return True, None
        
    except Exception as e:
        return False, str(e)

def train_model(model, dataloader, steps, device, freeze_except_position=False, lr=3e-5):
    """Train the model for specified number of steps."""
    
    # Test model compatibility first
    compatible, error = test_model_compatibility(model, dataloader, device)
    if not compatible:
        logger.error(f"Model compatibility test failed: {error}")
        raise RuntimeError(f"Model not compatible with current setup: {error}")
    
    if freeze_except_position:
        # Freeze all parameters except position embeddings
        for name, param in model.named_parameters():
            if "position_embeddings" not in name:
                param.requires_grad = False
        logger.info("Froze all parameters except position embeddings")
    else:
        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True
        logger.info("Training all parameters")
    
    # Setup optimizer and scheduler
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=min(500, steps // 10), 
        num_training_steps=steps
    )
    
    model.train()
    total_loss = 0.0
    step = 0
    
    while step < steps:
        for batch in dataloader:
            if step >= steps:
                break
                
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Check if batch has tokens to predict
            num_masked_tokens = (batch["labels"] != -100).sum().item()
            if num_masked_tokens == 0:
                continue
            
            try:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                
                loss = outputs.loss
                
                # Check for NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"NaN/inf loss at step {step}, skipping")
                    continue
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                step += 1
                
                # Log progress
                if step % 100 == 0:
                    avg_loss = total_loss / step
                    logger.info(f"Step {step}/{steps}, Average loss: {avg_loss:.4f}")
                    
            except Exception as e:
                logger.warning(f"Error in training step {step}: {str(e)}")
                continue
    
    avg_loss = total_loss / max(1, step)
    logger.info(f"Training completed. Average loss: {avg_loss:.4f}")
    return model

def run_experiment(experiment_name, model_size, corpus_file, seq_length, device, 
                  tokenizer, dataset=None, dataloader=None, max_eval_batches=1000):
    """Run a specific experiment from Table 5."""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Running experiment: {experiment_name} ({model_size})")
    logger.info(f"{'='*60}")
    
    if experiment_name == "roberta":
        # Experiment 1: RoBERTa (seqlen: 512)
        model = RobertaForMaskedLM.from_pretrained(f"roberta-{model_size}")
        model.to(device)
        
        # Test compatibility
        compatible, error = test_model_compatibility(model, dataloader, device)
        if not compatible:
            logger.error(f"RoBERTa model failed compatibility test: {error}")
            return float('nan')
        
        bpc = calculate_bpc(model, dataloader, device, max_batches=max_eval_batches)
        
    elif experiment_name == "longformer_no_copy":
        # Experiment 2: Longformer (seqlen: 4,096) without copy initialization
        roberta = RobertaForMaskedLM.from_pretrained(f"roberta-{model_size}")
        
        # Try to create Longformer with sliding window attention
        try:
            longformer_config = create_longformer_config(
                roberta.config, seq_length, attention_window=512  # As in paper
            )
            model = LongformerForMaskedLM.from_pretrained(f"roberta-{model_size}", config=longformer_config)
            model = extend_position_embeddings(model, seq_length, copy=False)
            model.to(device)
            
            # Test compatibility
            compatible, error = test_model_compatibility(model, dataloader, device)
            if not compatible:
                logger.error(f"Longformer model failed compatibility test: {error}")
                return float('nan')
            
            bpc = calculate_bpc(model, dataloader, device, max_batches=max_eval_batches)
            
        except Exception as e:
            logger.error(f"Failed to create/run Longformer model: {str(e)}")
            return float('nan')
        
    elif experiment_name == "longformer_copy":
        # Experiment 3: Longformer (seqlen: 4,096) with copy initialization
        roberta = RobertaForMaskedLM.from_pretrained(f"roberta-{model_size}")
        
        try:
            longformer_config = create_longformer_config(
                roberta.config, seq_length, attention_window=512  # As in paper
            )
            model = LongformerForMaskedLM.from_pretrained(f"roberta-{model_size}", config=longformer_config)
            model = extend_position_embeddings(model, seq_length, copy=True)
            model.to(device)
            
            # Test compatibility
            compatible, error = test_model_compatibility(model, dataloader, device)
            if not compatible:
                logger.error(f"Longformer model failed compatibility test: {error}")
                return float('nan')
            
            bpc = calculate_bpc(model, dataloader, device, max_batches=max_eval_batches)
            
        except Exception as e:
            logger.error(f"Failed to create/run Longformer model: {str(e)}")
            return float('nan')
        
    elif experiment_name == "longformer_copy_2k":
        # Experiment 4: Longformer with copy + 2K training steps
        roberta = RobertaForMaskedLM.from_pretrained(f"roberta-{model_size}")
        
        try:
            longformer_config = create_longformer_config(
                roberta.config, seq_length, attention_window=512  # As in paper
            )
            model = LongformerForMaskedLM.from_pretrained(f"roberta-{model_size}", config=longformer_config)
            model = extend_position_embeddings(model, seq_length, copy=True)
            model.to(device)
            
            # Calculate BPC before training
            bpc_before = calculate_bpc(model, dataloader, device, max_batches=100)
            logger.info(f"BPC before training: {bpc_before:.3f}")
            
            # Train for 2K steps
            model = train_model(model, dataloader, steps=2000, device=device)
            bpc = calculate_bpc(model, dataloader, device, max_batches=max_eval_batches)
            
        except Exception as e:
            logger.error(f"Failed to create/train Longformer model: {str(e)}")
            return float('nan')
        
    elif experiment_name == "longformer_copy_65k":
        # Experiment 5: Longformer with copy + 65K training steps
        roberta = RobertaForMaskedLM.from_pretrained(f"roberta-{model_size}")
        
        try:
            longformer_config = create_longformer_config(
                roberta.config, seq_length, attention_window=512  # As in paper
            )
            model = LongformerForMaskedLM.from_pretrained(f"roberta-{model_size}", config=longformer_config)
            model = extend_position_embeddings(model, seq_length, copy=True)
            model.to(device)
            
            # Calculate BPC before training
            bpc_before = calculate_bpc(model, dataloader, device, max_batches=100)
            logger.info(f"BPC before training: {bpc_before:.3f}")
            
            # Train for 65K steps
            model = train_model(model, dataloader, steps=65000, device=device)
            bpc = calculate_bpc(model, dataloader, device, max_batches=max_eval_batches)
            
        except Exception as e:
            logger.error(f"Failed to create/train Longformer model: {str(e)}")
            return float('nan')
        
    elif experiment_name == "longformer_freeze":
        # Experiment 6: Longformer (train extra pos. embed. only)
        roberta = RobertaForMaskedLM.from_pretrained(f"roberta-{model_size}")
        
        try:
            longformer_config = create_longformer_config(
                roberta.config, seq_length, attention_window=512  # As in paper
            )
            model = LongformerForMaskedLM.from_pretrained(f"roberta-{model_size}", config=longformer_config)
            model = extend_position_embeddings(model, seq_length, copy=True)
            model.to(device)
            
            # Calculate BPC before training
            bpc_before = calculate_bpc(model, dataloader, device, max_batches=100)
            logger.info(f"BPC before training: {bpc_before:.3f}")
            
            # Train only position embeddings for 2K steps
            model = train_model(model, dataloader, steps=2000, device=device, freeze_except_position=True)
            bpc = calculate_bpc(model, dataloader, device, max_batches=max_eval_batches)
            
        except Exception as e:
            logger.error(f"Failed to create/train Longformer model: {str(e)}")
            return float('nan')
        
    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")
    
    logger.info(f"Experiment {experiment_name} ({model_size}): BPC = {bpc:.3f}")
    return bpc

def main():
    parser = argparse.ArgumentParser(description="Replicate Table 5 with correct attention modes")
    parser.add_argument("--corpus", type=str, default="your_corpus.txt", help="Path to corpus file")
    parser.add_argument("--model_size", type=str, choices=["base", "large"], default="base", help="Model size")
    parser.add_argument("--experiments", type=str, nargs="+", 
                       choices=["roberta", "longformer_no_copy", "longformer_copy", 
                               "longformer_copy_2k", "longformer_copy_65k", "longformer_freeze", "all"],
                       default=["all"], help="Which experiments to run")
    parser.add_argument("--seq_length", type=int, default=4096, help="Sequence length for Longformer")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--max_eval_batches", type=int, default=1000, help="Max batches for evaluation")
    parser.add_argument("--output_file", type=str, default="table5_results.json", help="Output file for results")
    args = parser.parse_args()
    
    # Set seed
    set_seed(42)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(f"roberta-{args.model_size}")
    
    # Define all experiments
    all_experiments = [
        "roberta", 
        "longformer_no_copy", 
        "longformer_copy",
        "longformer_copy_2k", 
        "longformer_copy_65k", 
        "longformer_freeze"
    ]
    
    # Determine which experiments to run
    if "all" in args.experiments:
        experiments_to_run = all_experiments
    else:
        experiments_to_run = args.experiments
    
    # Prepare datasets for different sequence lengths
    datasets = {}
    dataloaders = {}
    
    # RoBERTa dataset (seq_length=512)
    if "roberta" in experiments_to_run:
        dataset_512 = MLMDataset(args.corpus, tokenizer, 512 - 2)
        datasets[512] = dataset_512
        dataloaders[512] = DataLoader(dataset_512, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Longformer dataset (seq_length=4096 or custom)
    if any(exp.startswith("longformer") for exp in experiments_to_run):
        dataset_long = MLMDataset(args.corpus, tokenizer, args.seq_length - 2)
        datasets[args.seq_length] = dataset_long
        dataloaders[args.seq_length] = DataLoader(dataset_long, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    # Run experiments
    results = {}
    
    for experiment in experiments_to_run:
        try:
            # Choose appropriate dataset
            if experiment == "roberta":
                seq_len = 512
            else:
                seq_len = args.seq_length
            
            dataset = datasets[seq_len]
            dataloader = dataloaders[seq_len]
            
            # Run experiment
            bpc = run_experiment(
                experiment, args.model_size, args.corpus, seq_len, device,
                tokenizer, dataset, dataloader, args.max_eval_batches
            )
            
            results[f"{experiment}_{args.model_size}"] = {
                "bpc": bpc,
                "seq_length": seq_len,
                "experiment": experiment,
                "model_size": args.model_size
            }
            
        except Exception as e:
            logger.error(f"Error running experiment {experiment}: {str(e)}")
            results[f"{experiment}_{args.model_size}"] = {
                "bpc": float('nan'),
                "error": str(e),
                "experiment": experiment,
                "model_size": args.model_size
            }
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary in Table 5 format
    logger.info(f"\n{'='*80}")
    logger.info("RESULTS SUMMARY - Table 5 Format")
    logger.info(f"{'='*80}")
    logger.info(f"{'Model':<35} {'base':<10} {'large':<10}")
    logger.info(f"{'-'*80}")
    
    # Print results in paper format
    experiment_labels = {
        "roberta": "RoBERTa (seqlen: 512)",
        "longformer_no_copy": "Longformer (seqlen: 4,096)",
        "longformer_copy": "+ copy position embeddings",
        "longformer_copy_2k": "+ 2K gradient updates", 
        "longformer_copy_65k": "+ 65K gradient updates",
        "longformer_freeze": "Longformer (train extra pos. embed. only)"
    }
    
    for exp in all_experiments:
        label = experiment_labels[exp]
        base_key = f"{exp}_base"
        large_key = f"{exp}_large"
        
        base_bpc = results.get(base_key, {}).get('bpc', 'N/A')
        large_bpc = results.get(large_key, {}).get('bpc', 'N/A')
        
        if isinstance(base_bpc, float) and not np.isnan(base_bpc):
            base_str = f"{base_bpc:.3f}"
        else:
            base_str = "FAILED"
            
        if isinstance(large_bpc, float) and not np.isnan(large_bpc):
            large_str = f"{large_bpc:.3f}"
        else:
            large_str = "FAILED"
        
        logger.info(f"{label:<35} {base_str:<10} {large_str:<10}")
    
    logger.info(f"\nDetailed results saved to {args.output_file}")

if __name__ == "__main__":
    main()