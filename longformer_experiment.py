import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import numpy as np
import os
import argparse
from tqdm import tqdm

class MicroLongformerConfig:
    """Ultra-small configuration for demo purposes only."""
    def __init__(
        self,
        hidden_size=64,          # Very small hidden size
        num_hidden_layers=3,     # Only 3 layers
        num_attention_heads=4,   # 4 attention heads (to allow dilation on subset)
        intermediate_size=128,   # Very small FF size
        attention_window=None,
        attention_dilation=None,
        max_position_embeddings=1024,  # Increased sequence length for more realistic BPC
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        dilation_on_heads=0,     # Number of heads to apply dilation to
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.attention_window = attention_window if attention_window else [16] * num_hidden_layers
        self.attention_dilation = attention_dilation if attention_dilation else [1] * num_hidden_layers
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.dilation_on_heads = dilation_on_heads

class FastLocalAttention(nn.Module):
    """Ultra-fast local attention implementation."""
    def __init__(self, config, layer_id):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.embed_dim = config.hidden_size
        
        self.query = nn.Linear(config.hidden_size, self.embed_dim)
        self.key = nn.Linear(config.hidden_size, self.embed_dim)
        self.value = nn.Linear(config.hidden_size, self.embed_dim)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.window_size = config.attention_window[layer_id]
        self.dilation = config.attention_dilation[layer_id]
        self.dilation_on_heads = config.dilation_on_heads
        
    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project inputs
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        
        # Reshape to heads
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scale query
        query = query / math.sqrt(self.head_dim)
        
        # Simple but faster approximation - just process a few chunks
        # For demo purposes, we don't need an exact implementation
        # We'll just compute regular attention on chunks of the sequence
        chunk_size = min(64, seq_len)  # Increased chunk size for efficiency
        num_chunks = math.ceil(seq_len / chunk_size)
        
        context_chunks = []
        for i in range(num_chunks):
            # Get chunk boundaries
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, seq_len)
            
            # Extract query for this chunk
            q_chunk = query[:, :, start_idx:end_idx, :]
            
            # Create context tensor for this chunk
            chunk_context = torch.zeros(
                batch_size, self.num_heads, end_idx - start_idx, self.head_dim, 
                device=hidden_states.device
            )
            
            # Process each head separately to apply dilation properly
            for h in range(self.num_heads):
                # Apply dilation only to specified number of heads
                effective_dilation = self.dilation if h < self.dilation_on_heads else 1
                effective_window = self.window_size if h < self.dilation_on_heads else self.window_size * self.dilation
                
                # Calculate window boundaries with dilation
                window_start = max(0, start_idx - effective_window * effective_dilation)
                window_end = min(seq_len, end_idx + effective_window * effective_dilation)
                
                # Extract key and value for this window
                k_window = key[:, h:h+1, window_start:window_end, :]
                v_window = value[:, h:h+1, window_start:window_end, :]
                
                # Compute attention scores
                attn_scores = torch.matmul(q_chunk[:, h:h+1], k_window.transpose(-1, -2))
                
                # Apply softmax to get attention probabilities
                attn_probs = F.softmax(attn_scores, dim=-1)
                
                # Apply attention to values
                head_context = torch.matmul(attn_probs, v_window)
                
                # Add to context tensor
                chunk_context[:, h:h+1] = head_context
            
            # Add to output chunks
            context_chunks.append(chunk_context)
        
        # Concatenate chunks
        if len(context_chunks) > 1:
            context = torch.cat(context_chunks, dim=2)
        else:
            context = context_chunks[0]
        
        # Transpose and reshape
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Output projection
        output = self.output(context)
        
        return output

class MicroLongformerLayer(nn.Module):
    """Simplified Transformer layer."""
    def __init__(self, config, layer_id):
        super().__init__()
        self.attention = FastLocalAttention(config, layer_id)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size)
        )
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states):
        # Self-attention block
        attention_output = self.attention(hidden_states)
        hidden_states = self.norm1(hidden_states + attention_output)
        
        # Feed-forward block
        ffn_output = self.ffn(hidden_states)
        hidden_states = self.norm2(hidden_states + self.dropout(ffn_output))
        
        return hidden_states

class MicroLongformer(nn.Module):
    """Ultra-simplified Longformer for quick experiments."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Character embeddings (ASCII)
        self.char_embeddings = nn.Embedding(256, config.hidden_size)
        
        # Position embeddings
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            MicroLongformerLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        
        # Output layer for LM
        self.lm_head = nn.Linear(config.hidden_size, 256)
        
        # Dropout and normalization
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.norm = nn.LayerNorm(config.hidden_size)
        
        self.init_weights()
        
    def init_weights(self):
        # Initialize weights
        nn.init.normal_(self.char_embeddings.weight, std=0.02)
        nn.init.normal_(self.position_embeddings.weight, std=0.02)
                    
    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.size()
        
        # Position ids
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        embeddings = self.char_embeddings(input_ids) + self.position_embeddings(position_ids)
        hidden_states = self.dropout(embeddings)
        
        # Transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        # LM prediction head
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift prediction logits and labels
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # Calculate cross entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, 256), shift_labels.view(-1))
        
        return logits, loss

class Text8Dataset:
    """Dataset for text8 with random batch generation to avoid memory issues."""
    def __init__(self, file_path, seq_length, split="train"):
        # Try to load the data
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            # If utf-8 fails, try latin-1
            with open(file_path, "r", encoding="latin-1") as f:
                text = f.read()
        except FileNotFoundError:
            print(f"Error: File {file_path} not found. Please check the path.")
            raise
        
        # Split data
        if split == "train":
            self.data = text[:int(0.9 * len(text))]
        elif split == "dev":
            self.data = text[int(0.9 * len(text)):int(0.95 * len(text))]
        elif split == "test":
            self.data = text[int(0.95 * len(text)):]
        
        self.seq_length = seq_length
        print(f"Loaded {split} data: {len(self.data)} characters")
    
    def get_random_batch(self, batch_size, device):
        """Generate a random batch from the dataset."""
        input_seqs = []
        target_seqs = []
        
        for _ in range(batch_size):
            # Pick a random starting point, leaving room for sequence plus one
            start_idx = np.random.randint(0, max(1, len(self.data) - self.seq_length - 1))
            
            # Extract sequence and target
            chunk = self.data[start_idx:start_idx + self.seq_length + 1]
            
            # Handle case where chunk is shorter than expected
            if len(chunk) < self.seq_length + 1:
                # Pad with spaces if needed
                chunk = chunk + ' ' * (self.seq_length + 1 - len(chunk))
            
            # Convert to list of character indices
            input_seq = [ord(c) % 256 for c in chunk[:-1]]  # Take all but last char
            target_seq = [ord(c) % 256 for c in chunk[1:]]  # Take all but first char
            
            input_seqs.append(input_seq)
            target_seqs.append(target_seq)
        
        # Convert to tensors
        input_ids = torch.tensor(input_seqs, dtype=torch.long, device=device)
        labels = torch.tensor(target_seqs, dtype=torch.long, device=device)
        
        return input_ids, labels

def ultra_fast_experiment(config_name, data_path, device, batch_size=4, steps=1000):
    """Run an ultra-fast experiment with the specified configuration."""
    print(f"Running experiment with configuration: {config_name}")
    
    # Create appropriate config based on window pattern
    if config_name == "decreasing":
        config = MicroLongformerConfig(
            attention_window=[128, 64, 32],  # Decreasing window sizes
            attention_dilation=[1, 1, 1],   # No dilation
            dilation_on_heads=0             # No dilation on any heads
        )
    elif config_name == "fixed":
        config = MicroLongformerConfig(
            attention_window=[64, 64, 64],  # Fixed window size
            attention_dilation=[1, 1, 1],   # No dilation
            dilation_on_heads=0             # No dilation on any heads
        )
    elif config_name == "increasing":
        config = MicroLongformerConfig(
            attention_window=[32, 64, 128],  # Increasing window sizes
            attention_dilation=[1, 1, 1],   # No dilation
            dilation_on_heads=0             # No dilation on any heads
        )
    elif config_name == "no_dilation":
        config = MicroLongformerConfig(
            attention_window=[32, 64, 128],  # Same as increasing
            attention_dilation=[1, 1, 1],   # No dilation
            dilation_on_heads=0             # No dilation on any heads
        )
    elif config_name == "dilation_2_heads":
        config = MicroLongformerConfig(
            attention_window=[32, 64, 128],  # Same as increasing
            attention_dilation=[1, 1, 2],   # Dilation on last layer
            dilation_on_heads=2             # Apply dilation to 2 heads
        )
    else:
        raise ValueError(f"Unknown configuration: {config_name}")
    
    # Create model and move to device
    model = MicroLongformer(config)
    model.to(device)
    
    # Print model stats
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model has {param_count:,} parameters")
    print(f"Window sizes: {config.attention_window}")
    print(f"Dilation: {config.attention_dilation}")
    print(f"Dilation on {config.dilation_on_heads} heads")
    
    # Load datasets
    train_dataset = Text8Dataset(data_path, config.max_position_embeddings, "train")
    dev_dataset = Text8Dataset(data_path, config.max_position_embeddings, "dev")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    model.train()
    start_time = time.time()
    
    # Create progress bar 
    progress_bar = tqdm(range(steps), desc=f"Training {config_name}")
    loss_values = []
    
    for step in progress_bar:
        # Get random batch
        input_ids, labels = train_dataset.get_random_batch(batch_size, device)
        
        # Forward and backward pass
        optimizer.zero_grad()
        _, loss = model(input_ids, labels=labels)
        loss.backward()
        optimizer.step()
        
        # Store loss
        loss_values.append(loss.item())
        
        # Update progress bar description instead of printing
        if (step + 1) % 100 == 0:
            elapsed = time.time() - start_time
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "Time": f"{elapsed:.2f}s"})
    
    # Evaluate more thoroughly with corrected calculation
    print("\nEvaluating model...")
    model.eval()
    total_loss = 0
    total_chars = 0
    num_batches = 20  # More evaluation batches for better accuracy
    
    with torch.no_grad():
        for _ in range(num_batches):
            input_ids, labels = dev_dataset.get_random_batch(batch_size, device)
            _, loss = model(input_ids, labels=labels)
            
            # Count loss per character, accounting for shifted sequences
            # CrossEntropyLoss returns mean loss per token
            # Each sequence has seq_len-1 tokens due to shifting
            actual_tokens = input_ids.size(0) * (input_ids.size(1) - 1)
            total_loss += loss.item() * actual_tokens
            total_chars += actual_tokens
    
    # Calculate BPC (bits per character) with corrected count
    avg_loss = total_loss / total_chars
    bpc = avg_loss / math.log(2)  # Convert from nats to bits
    
    # Print results
    print(f"Configuration: {config_name}")
    print(f"BPC: {bpc:.4f}")
    
    return bpc

def main():
    parser = argparse.ArgumentParser(description="Run ultra-fast Longformer Table 4 experiments")
    parser.add_argument(
        "--config", 
        type=str, 
        default="all", 
        choices=["all", "decreasing", "fixed", "increasing", "no_dilation", "dilation_2_heads"],
        help="Which window configuration to test"
    )
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="./data/text8",
        help="Path to text8 dataset"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4,  # Reduced batch size for longer sequences
        help="Batch size for training"
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=1000,  # Increased steps
        help="Number of training steps"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./results",
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    
    # Check if text8 dataset exists
    if not os.path.exists(args.data_path):
        print(f"Text8 dataset not found at {args.data_path}")
        print("Checking if the dataset needs to be downloaded...")
        
        # Create data directory if needed
        os.makedirs(os.path.dirname(args.data_path), exist_ok=True)
        
        # Download dataset
        if not os.path.exists(args.data_path):
            try:
                import urllib.request
                import zipfile
                
                print("Downloading text8 dataset...")
                urllib.request.urlretrieve("http://mattmahoney.net/dc/text8.zip", "data/text8.zip")
                
                print("Extracting text8 dataset...")
                with zipfile.ZipFile("data/text8.zip", "r") as zip_ref:
                    zip_ref.extractall("data")
                
                print(f"Dataset downloaded and extracted to {args.data_path}")
            except Exception as e:
                print(f"Error downloading dataset: {e}")
                print("Please download text8 dataset manually from http://mattmahoney.net/dc/text8.zip")
                return
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run experiments
    if args.config == "all":
        configs = ["decreasing", "fixed", "increasing", "no_dilation", "dilation_2_heads"]
        results = {}
        
        for config in configs:
            bpc = ultra_fast_experiment(config, args.data_path, device, args.batch_size, args.steps)
            results[config] = bpc
        
        # Write results to file
        with open(os.path.join(args.output_dir, "longformer_results.txt"), "w") as f:
            f.write("Results from our implementation with corrected evaluation:\n\n")
            for config, bpc in results.items():
                f.write(f"{config}: {bpc:.4f}\n")
        
        # Print summary and compare with original paper
        print("\n" + "="*60)
        print("RESULTS SUMMARY:")
        print("="*60)
        for config, bpc in results.items():
            print(f"{config}: {bpc:.4f}")
        
        # Compare with original paper results
        print("\n" + "="*60)
        print("COMPARISON WITH ORIGINAL LONGFORMER PAPER (Table 4):")
        print("="*60)
        print("Original paper results (after 150K steps):")
        print("Decreasing window (512 to 32): 1.24 BPC")
        print("Fixed window (= 230): 1.23 BPC")
        print("Increasing window (32 to 512): 1.21 BPC")
        print("No Dilation: 1.21 BPC")
        print("Dilation on 2 heads: 1.20 BPC")
        
        # Get ranking of configurations
        sorted_configs = sorted(results.items(), key=lambda x: x[1])
        
        print("\nRanking of configurations (best to worst):")
        for i, (config, bpc) in enumerate(sorted_configs):
            print(f"{i+1}. {config}: {bpc:.4f}")
        
        # Check if our results match the paper's pattern
        if (sorted_configs[0][0] == "dilation_2_heads" and 
            sorted_configs[-1][0] == "decreasing"):
            print("\nOur results match the pattern from the original paper:")
            print("- Dilation on 2 heads performs best")
            print("- Decreasing window sizes performs worst")
        else:
            print("\nOur results show a different pattern than the original paper.")
    else:
        bpc = ultra_fast_experiment(args.config, args.data_path, device, args.batch_size, args.steps)
        
        # Write result to file
        with open(os.path.join(args.output_dir, f"{args.config}_result.txt"), "w") as f:
            f.write(f"BPC: {bpc:.4f}\n")

if __name__ == "__main__":
    main()