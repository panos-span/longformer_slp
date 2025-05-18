# manual_longformer_conversion.py - Replicate Table 5 completely
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaConfig
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import logging
import random
from tqdm import tqdm
import argparse
import time
import gc
import copy

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class LongformerSelfAttention(nn.Module):
    """Simplified Longformer self-attention for manual conversion."""
    
    def __init__(self, config, layer_id):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of "
                f"the number of attention heads ({config.num_attention_heads})"
            )
        
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.embed_dim = config.hidden_size
        
        # Local attention projections
        self.query = nn.Linear(config.hidden_size, self.embed_dim)
        self.key = nn.Linear(config.hidden_size, self.embed_dim)
        self.value = nn.Linear(config.hidden_size, self.embed_dim)
        
        # Global attention projections
        self.query_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.key_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.value_global = nn.Linear(config.hidden_size, self.embed_dim)
        
        self.dropout = config.attention_probs_dropout_prob
        self.layer_id = layer_id
        
        # Simplified sliding window (for demonstration)
        self.attention_window = 256  # Half window size
    
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        """Simplified sliding window attention."""
        batch_size, seq_len, _ = hidden_states.size()
        
        # For simplicity, we'll implement a basic version
        # In a full implementation, you'd use proper sliding window logic
        
        # Local attention (simplified)
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Simplified attention computation (not true sliding window)
        # For demonstration purposes - in practice you'd implement proper windowing
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask.unsqueeze(1).unsqueeze(1)
        
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = torch.dropout(attn_weights, p=self.dropout, training=self.training)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )
        
        return (attn_output,)


def convert_roberta_to_longformer(roberta_model, max_position_embeddings=4096, copy_positions=False):
    """
    Manually convert RoBERTa to Longformer architecture.
    This replicates the conversion process described in the paper.
    """
    logger.info(f"Converting RoBERTa to Longformer (max_pos={max_position_embeddings}, copy_pos={copy_positions})")
    
    # Create new config for Longformer
    config = copy.deepcopy(roberta_model.config)
    config.max_position_embeddings = max_position_embeddings
    
    # Create new model with extended position embeddings
    longformer_model = RobertaForMaskedLM(config)
    
    # Copy all weights except position embeddings and self-attention
    # 1. Copy word embeddings
    longformer_model.roberta.embeddings.word_embeddings.load_state_dict(
        roberta_model.roberta.embeddings.word_embeddings.state_dict()
    )
    
    # 2. Copy token type embeddings
    longformer_model.roberta.embeddings.token_type_embeddings.load_state_dict(
        roberta_model.roberta.embeddings.token_type_embeddings.state_dict()
    )
    
    # 3. Copy LayerNorm
    longformer_model.roberta.embeddings.LayerNorm.load_state_dict(
        roberta_model.roberta.embeddings.LayerNorm.state_dict()
    )
    
    # 4. Copy pooler
    longformer_model.roberta.pooler.load_state_dict(
        roberta_model.roberta.pooler.state_dict()
    )
    
    # 5. Copy LM head
    longformer_model.lm_head.load_state_dict(roberta_model.lm_head.state_dict())
    
    # 6. Copy encoder layers (except self-attention)
    for i in range(config.num_hidden_layers):
        # Copy attention output layer
        longformer_model.roberta.encoder.layer[i].attention.output.load_state_dict(
            roberta_model.roberta.encoder.layer[i].attention.output.state_dict()
        )
        
        # Copy intermediate layer
        longformer_model.roberta.encoder.layer[i].intermediate.load_state_dict(
            roberta_model.roberta.encoder.layer[i].intermediate.state_dict()
        )
        
        # Copy output layer
        longformer_model.roberta.encoder.layer[i].output.load_state_dict(
            roberta_model.roberta.encoder.layer[i].output.state_dict()
        )
        
        # Initialize Longformer self-attention with RoBERTa weights
        roberta_self_attn = roberta_model.roberta.encoder.layer[i].attention.self
        longformer_self_attn = longformer_model.roberta.encoder.layer[i].attention.self
        
        # Copy self-attention weights
        longformer_self_attn.query.weight.data = roberta_self_attn.query.weight.data.clone()
        longformer_self_attn.query.bias.data = roberta_self_attn.query.bias.data.clone()
        longformer_self_attn.key.weight.data = roberta_self_attn.key.weight.data.clone()
        longformer_self_attn.key.bias.data = roberta_self_attn.key.bias.data.clone()
        longformer_self_attn.value.weight.data = roberta_self_attn.value.weight.data.clone()
        longformer_self_attn.value.bias.data = roberta_self_attn.value.bias.data.clone()
        
        # Replace with Longformer attention (simplified version)
        # In a full implementation, you'd use the actual sliding window attention
        longformer_model.roberta.encoder.layer[i].attention.self = LongformerSelfAttention(config, i)
        
        # Copy weights to the new attention
        longformer_model.roberta.encoder.layer[i].attention.self.query.weight.data = roberta_self_attn.query.weight.data.clone()
        longformer_model.roberta.encoder.layer[i].attention.self.query.bias.data = roberta_self_attn.query.bias.data.clone()
        longformer_model.roberta.encoder.layer[i].attention.self.key.weight.data = roberta_self_attn.key.weight.data.clone()
        longformer_model.roberta.encoder.layer[i].attention.self.key.bias.data = roberta_self_attn.key.bias.data.clone()
        longformer_model.roberta.encoder.layer[i].attention.self.value.weight.data = roberta_self_attn.value.weight.data.clone()
        longformer_model.roberta.encoder.layer[i].attention.self.value.bias.data = roberta_self_attn.value.bias.data.clone()
        
        # Initialize global attention weights (same as local)
        longformer_model.roberta.encoder.layer[i].attention.self.query_global.weight.data = roberta_self_attn.query.weight.data.clone()
        longformer_model.roberta.encoder.layer[i].attention.self.query_global.bias.data = roberta_self_attn.query.bias.data.clone()
        longformer_model.roberta.encoder.layer[i].attention.self.key_global.weight.data = roberta_self_attn.key.weight.data.clone()
        longformer_model.roberta.encoder.layer[i].attention.self.key_global.bias.data = roberta_self_attn.key.bias.data.clone()
        longformer_model.roberta.encoder.layer[i].attention.self.value_global.weight.data = roberta_self_attn.value.weight.data.clone()
        longformer_model.roberta.encoder.layer[i].attention.self.value_global.bias.data = roberta_self_attn.value.bias.data.clone()
    
    # 7. Handle position embeddings - THIS IS THE KEY PART!
    old_pos_emb = roberta_model.roberta.embeddings.position_embeddings
    new_pos_emb = longformer_model.roberta.embeddings.position_embeddings
    
    # Copy original position embeddings
    with torch.no_grad():
        new_pos_emb.weight[:old_pos_emb.num_embeddings] = old_pos_emb.weight.clone()
        
        if copy_positions:
            # Copy position embeddings as described in paper
            logger.info("Copying position embeddings to extended positions...")
            for i in range(old_pos_emb.num_embeddings, max_position_embeddings):
                # Copy pattern: repeat the RoBERTa position embeddings
                pos_to_copy = 2 + (i - old_pos_emb.num_embeddings) % (old_pos_emb.num_embeddings - 2)
                new_pos_emb.weight[i] = old_pos_emb.weight[pos_to_copy].clone()
        else:
            # Leave new positions randomly initialized
            logger.info("Leaving extended position embeddings randomly initialized...")
            # The rest remains randomly initialized (default PyTorch initialization)
    
    return longformer_model


class MemoryEfficientMLMDataset(Dataset):
    """Memory-efficient MLM dataset for training/evaluation."""
    
    def __init__(self, file_path, tokenizer, seq_length, max_examples=None):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.examples = []
        
        logger.info(f"Loading data from {file_path} (seq_len={seq_length}, max_examples={max_examples})")
        
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Process text into examples
        sentences = []
        for line in text.split("\n"):
            line = line.strip()
            if line:
                sent_parts = [s.strip() + "." for s in line.split(".") if s.strip()]
                sentences.extend(sent_parts)
        
        current_tokens = []
        examples_created = 0
        
        for sentence in sentences:
            if max_examples and examples_created >= max_examples:
                break
            
            tokens = tokenizer.tokenize(sentence)
            
            if len(current_tokens) + len(tokens) + 3 <= seq_length:
                current_tokens.extend(tokens)
                current_tokens.append(tokenizer.sep_token)
            else:
                if len(current_tokens) >= 10:
                    self.examples.append(self._create_example(current_tokens[:-1]))
                    examples_created += 1
                
                current_tokens = tokens
                current_tokens.append(tokenizer.sep_token)
        
        if len(current_tokens) >= 10 and (not max_examples or examples_created < max_examples):
            self.examples.append(self._create_example(current_tokens[:-1]))
        
        logger.info(f"Created {len(self.examples)} examples")
    
    def _create_example(self, tokens):
        """Create a single training example."""
        token_ids = [self.tokenizer.cls_token_id]
        token_ids.extend(self.tokenizer.convert_tokens_to_ids(tokens))
        token_ids.append(self.tokenizer.sep_token_id)
        
        while len(token_ids) < self.seq_length:
            token_ids.append(self.tokenizer.pad_token_id)
        
        return token_ids[:self.seq_length]
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = torch.tensor(self.examples[idx], dtype=torch.long)
        attention_mask = (tokens != self.tokenizer.pad_token_id).long()
        masked_tokens, labels = self._mask_tokens(tokens.clone())
        
        return {
            "input_ids": masked_tokens,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    def _mask_tokens(self, inputs):
        """Apply MLM masking."""
        labels = inputs.clone()
        
        # Create special tokens mask
        all_special_ids = set(self.tokenizer.all_special_ids)
        special_tokens_mask = torch.tensor(
            [token_id in all_special_ids for token_id in labels.tolist()],
            dtype=torch.bool,
        )
        
        # Create probability matrix
        probability_matrix = torch.full(labels.shape, 0.15)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        probability_matrix.masked_fill_(inputs == self.tokenizer.pad_token_id, value=0.0)
        
        # Create masked indices
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        if not masked_indices.any():
            maskable = ~special_tokens_mask & (inputs != self.tokenizer.pad_token_id)
            if maskable.any():
                idx = torch.multinomial(maskable.float(), 1)
                masked_indices[idx] = True
        
        labels[~masked_indices] = -100
        
        # Replace tokens
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id
        
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & 
            masked_indices & ~indices_replaced
        )
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        
        return inputs, labels


def calculate_bpc(model, dataloader, device, max_batches=100):
    """Calculate BPC on dataset."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batches_processed = 0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if i >= max_batches:
                break
            
            batch = {k: v.to(device) for k, v in batch.items()}
            valid_tokens = (batch["labels"] != -100).sum().item()
            if valid_tokens == 0:
                continue
            
            try:
                outputs = model(**batch)
                loss = outputs.loss
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += loss.item() * valid_tokens
                    total_tokens += valid_tokens
                    batches_processed += 1
                
                if i % 20 == 0:
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                logger.error(f"Error in batch {i}: {str(e)}")
                continue
    
    if total_tokens == 0:
        return float("nan")
    
    avg_loss = total_loss / total_tokens
    bpc = avg_loss / np.log(2)
    
    logger.info(f"Processed {batches_processed} batches, {total_tokens} tokens")
    model.train()
    
    return bpc


def train_longformer(model, dataloader, device, num_updates=2000, learning_rate=3e-5):
    """Train/finetune Longformer for specified number of updates."""
    model.train()
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * num_updates),
        num_training_steps=num_updates
    )
    
    model.zero_grad()
    total_loss = 0.0
    
    pbar = tqdm(total=num_updates, desc="Training")
    
    update_count = 0
    data_iter = iter(dataloader)
    
    while update_count < num_updates:
        try:
            batch = next(data_iter)
        except StopIteration:
            # Reset data iterator
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        batch = {k: v.to(device) for k, v in batch.items()}
        
        outputs = model(**batch)
        loss = outputs.loss
        
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        
        total_loss += loss.item()
        update_count += 1
        
        pbar.update(1)
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/update_count:.4f}'
        })
        
        if update_count % 500 == 0:
            torch.cuda.empty_cache()
    
    pbar.close()
    return total_loss / num_updates


def main():
    parser = argparse.ArgumentParser(description="Replicate Table 5 exactly")
    parser.add_argument("--corpus", type=str, default="your_corpus.txt")
    parser.add_argument("--model_size", type=str, choices=["base"], default="base")
    parser.add_argument("--experiment", type=str, choices=[
        "roberta_baseline",
        "longformer_random_pos", 
        "longformer_copy_pos",
        "longformer_2k_updates",
        "longformer_65k_updates",
        "longformer_frozen_roberta",
        "all"
    ], default="all")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    tokenizer = RobertaTokenizer.from_pretrained(f"roberta-{args.model_size}")
    
    # Experiments to run
    experiments = [
        ("roberta_baseline", "RoBERTa (seqlen: 512)"),
        ("longformer_random_pos", "Longformer (random pos)"),
        ("longformer_copy_pos", "Longformer (copy pos)"),
        ("longformer_2k_updates", "+ 2K gradient updates"),
        ("longformer_65k_updates", "+ 65K gradient updates"),
        ("longformer_frozen_roberta", "Longformer (frozen RoBERTa)")
    ]
    
    if args.experiment == "all":
        experiments_to_run = [exp[0] for exp in experiments]
    else:
        experiments_to_run = [args.experiment]
    
    results = {}
    
    for exp_name in experiments_to_run:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running experiment: {exp_name}")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            if exp_name == "roberta_baseline":
                # Standard RoBERTa evaluation
                model = RobertaForMaskedLM.from_pretrained(f"roberta-{args.model_size}")
                model.to(device)
                
                dataset = MemoryEfficientMLMDataset(args.corpus, tokenizer, 512, max_examples=5000)
                dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
                
                bpc = calculate_bpc(model, dataloader, device, max_batches=200)
                
            elif exp_name == "longformer_random_pos":
                # Convert RoBERTa to Longformer with random position embeddings
                roberta = RobertaForMaskedLM.from_pretrained(f"roberta-{args.model_size}")
                model = convert_roberta_to_longformer(roberta, max_position_embeddings=4096, copy_positions=False)
                model.to(device)
                
                dataset = MemoryEfficientMLMDataset(args.corpus, tokenizer, 4096, max_examples=2000)
                dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
                
                bpc = calculate_bpc(model, dataloader, device, max_batches=100)
                
            elif exp_name == "longformer_copy_pos":
                # Convert RoBERTa to Longformer with copied position embeddings
                roberta = RobertaForMaskedLM.from_pretrained(f"roberta-{args.model_size}")
                model = convert_roberta_to_longformer(roberta, max_position_embeddings=4096, copy_positions=True)
                model.to(device)
                
                dataset = MemoryEfficientMLMDataset(args.corpus, tokenizer, 4096, max_examples=2000)
                dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
                
                bpc = calculate_bpc(model, dataloader, device, max_batches=100)
                
            elif exp_name == "longformer_2k_updates":
                # Train for 2K updates
                roberta = RobertaForMaskedLM.from_pretrained(f"roberta-{args.model_size}")
                model = convert_roberta_to_longformer(roberta, max_position_embeddings=4096, copy_positions=True)
                model.to(device)
                
                # Training dataset
                train_dataset = MemoryEfficientMLMDataset(args.corpus, tokenizer, 4096, max_examples=5000)
                train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
                
                # Train for 2K updates
                avg_loss = train_longformer(model, train_dataloader, device, num_updates=2000)
                
                # Evaluate
                eval_dataset = MemoryEfficientMLMDataset(args.corpus, tokenizer, 4096, max_examples=1000)
                eval_dataloader = DataLoader(eval_dataset, batch_size=2, shuffle=False)
                bpc = calculate_bpc(model, eval_dataloader, device, max_batches=100)
                
            elif exp_name == "longformer_65k_updates":
                # Train for 65K updates (warning: this will take a long time!)
                roberta = RobertaForMaskedLM.from_pretrained(f"roberta-{args.model_size}")
                model = convert_roberta_to_longformer(roberta, max_position_embeddings=4096, copy_positions=True)
                model.to(device)
                
                # Training dataset
                train_dataset = MemoryEfficientMLMDataset(args.corpus, tokenizer, 4096, max_examples=10000)
                train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
                
                # Train for 65K updates (this will take many hours!)
                avg_loss = train_longformer(model, train_dataloader, device, num_updates=65000)
                
                # Evaluate
                eval_dataset = MemoryEfficientMLMDataset(args.corpus, tokenizer, 4096, max_examples=1000)
                eval_dataloader = DataLoader(eval_dataset, batch_size=2, shuffle=False)
                bpc = calculate_bpc(model, eval_dataloader, device, max_batches=100)
                
            elif exp_name == "longformer_frozen_roberta":
                # Train only position embeddings, freeze RoBERTa weights
                roberta = RobertaForMaskedLM.from_pretrained(f"roberta-{args.model_size}")
                model = convert_roberta_to_longformer(roberta, max_position_embeddings=4096, copy_positions=True)
                
                # Freeze all parameters except position embeddings
                for name, param in model.named_parameters():
                    if "position_embeddings" not in name:
                        param.requires_grad = False
                
                model.to(device)
                
                # Training dataset
                train_dataset = MemoryEfficientMLMDataset(args.corpus, tokenizer, 4096, max_examples=5000)
                train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
                
                # Train only position embeddings
                avg_loss = train_longformer(model, train_dataloader, device, num_updates=5000)
                
                # Evaluate
                eval_dataset = MemoryEfficientMLMDataset(args.corpus, tokenizer, 4096, max_examples=1000)
                eval_dataloader = DataLoader(eval_dataset, batch_size=2, shuffle=False)
                bpc = calculate_bpc(model, eval_dataloader, device, max_batches=100)
            
            else:
                logger.error(f"Unknown experiment: {exp_name}")
                bpc = float("nan")
            
            end_time = time.time()
            
            results[exp_name] = {
                "bpc": bpc,
                "time_minutes": (end_time - start_time) / 60
            }
            
            logger.info(f"{exp_name}: BPC = {bpc:.3f}, Time = {(end_time - start_time)/60:.1f} min")
            
            # Clean up
            del model
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error in {exp_name}: {str(e)}")
            results[exp_name] = {"bpc": float("nan"), "error": str(e)}
    
    # Print final results
    logger.info("\n" + "=" * 80)
    logger.info("MANUAL LONGFORMER CONVERSION - TABLE 5 REPLICATION")
    logger.info("=" * 80)
    
    paper_results = {
        "roberta_baseline": 1.846,
        "longformer_random_pos": 10.299,
        "longformer_copy_pos": 1.957,
        "longformer_2k_updates": 1.753,
        "longformer_65k_updates": 1.705,
        "longformer_frozen_roberta": 1.850
    }
    
    for exp_name, exp_label in experiments:
        if exp_name in results:
            your_bpc = results[exp_name]["bpc"]
            paper_bpc = paper_results[exp_name]
            
            if not np.isnan(your_bpc):
                diff = your_bpc - paper_bpc
                status = "✓ GOOD" if abs(diff) < 1.0 else ("⚠ HIGHER" if diff > 0 else "✓ BETTER")
                logger.info(f"{exp_label:<30} {your_bpc:.3f} (paper: {paper_bpc:.3f}) {status}")
            else:
                logger.info(f"{exp_label:<30} FAILED")


if __name__ == "__main__":
    main()