import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import (
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
    RobertaConfig,
)
from torch.optim import AdamW
import logging
import random
from tqdm import tqdm
import sys

# Add longformer module to path
sys.path.append(os.path.join(os.path.dirname(__file__), "longformer"))
from longformer import LongformerForMaskedLM, LongformerConfig

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(42)

# Config - REDUCED VALUES FOR MEMORY EFFICIENCY
MAX_SEQ_LENGTH = 512  # Reduced for 8GB GPU
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 32
LEARNING_RATE = 5e-5
WARMUP_STEPS = 500
TOTAL_STEPS = 20000
SAVE_EVERY = 5000
EVAL_EVERY = 1000
LOG_EVERY = 10


# MLM Dataset
class MLMDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_length):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.examples = []

        logger.info(f"Loading data from {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Split into documents
        docs = text.split("\n\n")
        logger.info(f"Found {len(docs)} documents")

        # Process each document
        for doc in tqdm(docs, desc="Processing documents"):
            if len(doc.strip()) > 0:
                tokens = tokenizer.tokenize(doc)
                token_ids = tokenizer.convert_tokens_to_ids(tokens)

                # Split into chunks of max_seq_length
                for i in range(0, len(token_ids), seq_length):
                    chunk = token_ids[i : i + seq_length]
                    if len(chunk) >= 16:  # Minimum length to avoid issues
                        # Pad if necessary
                        if len(chunk) < seq_length:
                            chunk = chunk + [tokenizer.pad_token_id] * (
                                seq_length - len(chunk)
                            )
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
            "labels": labels,
        }

    def mask_tokens(self, inputs):
        """Prepare masked tokens for MLM training."""
        labels = inputs.clone()

        # We sample a few tokens in each sequence for masked-LM training
        probability_matrix = torch.full(labels.shape, 0.15)

        # Create special tokens mask
        special_tokens_mask = torch.zeros_like(inputs, dtype=torch.bool)

        # Mark special tokens like [CLS], [SEP], [PAD], etc.
        special_token_ids = self.tokenizer.all_special_ids
        for special_id in special_token_ids:
            special_tokens_mask = special_tokens_mask | (inputs == special_id)

        # Don't mask special tokens
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # Create masked indices
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) keep the masked input tokens unchanged
        return inputs, labels


def extend_position_and_buffers(model, max_seq_length):
    """Extend position embeddings and internal buffers to support longer sequences."""
    # Add +2 for the special tokens
    max_pos = max_seq_length + 2

    # 1. Extend position embeddings
    current_max_pos = model.roberta.embeddings.position_embeddings.weight.size(0)

    if max_pos > current_max_pos:
        # Create new position embeddings
        current_emb = model.roberta.embeddings.position_embeddings.weight.data
        new_emb = current_emb.new_empty(max_pos, current_emb.size(1))

        # Copy old embeddings
        new_emb[:current_max_pos] = current_emb

        # Fill the rest with copies using a pattern similar to RoBERTa
        for i in range(current_max_pos, max_pos):
            # Simple repeating pattern, copying from position 2 onwards
            # (positions 0 and 1 are special)
            new_emb[i] = current_emb[2 + (i - current_max_pos) % (current_max_pos - 2)]

        # Update position embeddings
        model.roberta.embeddings.position_embeddings = (
            torch.nn.Embedding.from_pretrained(new_emb, freeze=False)
        )

        # Update position_ids - crucial for the model to handle longer sequences
        model.roberta.embeddings.register_buffer(
            "position_ids", torch.arange(max_pos).expand((1, -1))
        )

        # 2. Fix token_type_ids buffer
        # RoBERTa doesn't use token_type_ids but the buffer exists and needs to be resized
        if hasattr(model.roberta.embeddings, "token_type_ids"):
            model.roberta.embeddings.register_buffer(
                "token_type_ids", torch.zeros(1, max_pos, dtype=torch.long)
            )

        # Update max position in config
        model.config.max_position_embeddings = max_pos

        logger.info(f"Extended position embeddings from {current_max_pos} to {max_pos}")

    return model


def evaluate(model, eval_dataloader, device):
    """Run evaluation and return perplexity."""
    model.eval()
    total_loss = 0
    total_examples = 0

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            total_loss += outputs.loss.item() * batch["input_ids"].size(0)
            total_examples += batch["input_ids"].size(0)

    avg_loss = total_loss / total_examples
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    model.train()
    return perplexity


def main():
    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # Create LongformerConfig based on RobertaConfig but using 'n2' attention mode
    # This uses the original RoBERTa self-attention, which will be more compatible
    roberta_config = RobertaConfig.from_pretrained("roberta-base")

    # Create LongformerConfig with our settings
    config = LongformerConfig(
        # Copy all attributes from RobertaConfig
        vocab_size=roberta_config.vocab_size,
        hidden_size=roberta_config.hidden_size,
        num_hidden_layers=roberta_config.num_hidden_layers,
        num_attention_heads=roberta_config.num_attention_heads,
        intermediate_size=roberta_config.intermediate_size,
        hidden_act=roberta_config.hidden_act,
        hidden_dropout_prob=roberta_config.hidden_dropout_prob,
        attention_probs_dropout_prob=roberta_config.attention_probs_dropout_prob,
        max_position_embeddings=roberta_config.max_position_embeddings,
        type_vocab_size=roberta_config.type_vocab_size,
        initializer_range=roberta_config.initializer_range,
        layer_norm_eps=roberta_config.layer_norm_eps,
        # Longformer specific settings
        attention_window=[16] * roberta_config.num_hidden_layers,
        attention_dilation=[1] * roberta_config.num_hidden_layers,
        autoregressive=False,
        attention_mode="n2",  # Use n2 mode which is the original RoBERTa attention (no sliding window)
    )

    # Initialize the Longformer model with weights from RoBERTa
    logger.info("Initializing Longformer from RoBERTa with n2 attention mode")
    model = LongformerForMaskedLM.from_pretrained("roberta-base", config=config)

    # Extend position embeddings for longer sequences
    logger.info("Extending position embeddings")
    model = extend_position_and_buffers(model, MAX_SEQ_LENGTH)

    # Move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load dataset
    train_dataset = MLMDataset(
        file_path="your_corpus.txt",  # Replace with your corpus file
        tokenizer=tokenizer,
        seq_length=MAX_SEQ_LENGTH,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,  # Reduce if causing issues
        pin_memory=True,
    )

    # Create a small evaluation dataset
    eval_size = min(len(train_dataset) // 10, 1000)
    eval_indices = random.sample(range(len(train_dataset)), eval_size)
    eval_dataset = torch.utils.data.Subset(train_dataset, eval_indices)

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Prepare optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, eps=1e-6)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=TOTAL_STEPS
    )

    # GradScaler for mixed precision training
    scaler = torch.amp.GradScaler()

    # Training loop
    global_step = 0
    tr_loss = 0.0
    model.train()

    logger.info("Starting training")
    for epoch in range(10):  # Multiple epochs until we hit total steps
        epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(epoch_iterator):
            # Move batch to GPU
            batch = {k: v.to(device) for k, v in batch.items()}

            # With mixed precision training
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS

            scaler.scale(loss).backward()
            tr_loss += loss.item()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                scheduler.step()
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

                # Log progress
                if global_step % LOG_EVERY == 0:
                    logger.info(f"Step {global_step} - Loss: {tr_loss/LOG_EVERY:.4f}")
                    tr_loss = 0.0

                # Save checkpoint
                if global_step % SAVE_EVERY == 0:
                    output_dir = f"./model_checkpoint_{global_step}"
                    os.makedirs(output_dir, exist_ok=True)
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    logger.info(f"Saved model to {output_dir}")

                # Evaluate
                if global_step % EVAL_EVERY == 0:
                    perplexity = evaluate(model, eval_dataloader, device)
                    logger.info(f"Step {global_step} - Perplexity: {perplexity:.2f}")

                # Check if we've reached the total steps
                if global_step >= TOTAL_STEPS:
                    logger.info("Reached total steps. Stopping training.")
                    # Save the final model
                    output_dir = "./model_final"
                    os.makedirs(output_dir, exist_ok=True)
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    return

            # Check GPU memory usage periodically
            if step % 100 == 0:
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / 1024**2
                    logger.info(f"GPU Memory Usage: {gpu_memory:.2f} MB")


if __name__ == "__main__":
    main()
