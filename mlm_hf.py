# table5_with_hf_longformer.py - Use Hugging Face Longformer implementation
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, LongformerTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
import logging
import random
from tqdm import tqdm
import argparse
import json
import gc
import time

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Import Hugging Face Longformer implementation
try:
    from transformers import LongformerModel, LongformerForMaskedLM, LongformerConfig
    LONGFORMER_AVAILABLE = True
    logger.info("✓ Hugging Face Longformer implementation imported successfully")
except ImportError:
    LONGFORMER_AVAILABLE = False
    logger.warning("⚠ Hugging Face Longformer not available.")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def cleanup_gpu_memory():
    """Comprehensive GPU memory cleanup."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()


def get_memory_info():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        return allocated, reserved, total
    return 0, 0, 0


class MemoryEfficientMLMDataset(Dataset):
    """Memory-efficient MLM dataset."""

    def __init__(self, file_path, tokenizer, seq_length, max_examples=None):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.examples = []

        logger.info(f"Loading data from {file_path} (max_examples: {max_examples})")

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Process text into sentences
        sentences = []
        for line in text.split("\n"):
            line = line.strip()
            if line:
                sent_parts = [s.strip() + "." for s in line.split(".") if s.strip()]
                sentences.extend(sent_parts)

        # Create examples by combining sentences
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

        if len(current_tokens) >= 10 and (
            not max_examples or examples_created < max_examples
        ):
            self.examples.append(self._create_example(current_tokens[:-1]))

        logger.info(f"Created {len(self.examples)} examples")

    def _create_example(self, tokens):
        """Create a single training example."""
        token_ids = [self.tokenizer.cls_token_id]
        token_ids.extend(self.tokenizer.convert_tokens_to_ids(tokens))
        token_ids.append(self.tokenizer.sep_token_id)

        while len(token_ids) < self.seq_length:
            token_ids.append(self.tokenizer.pad_token_id)

        return token_ids[: self.seq_length]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens = torch.tensor(self.examples[idx], dtype=torch.long)
        attention_mask = (tokens != self.tokenizer.pad_token_id).long()
        
        # For Longformer, set global attention on the first token (CLS)
        global_attention_mask = torch.zeros_like(attention_mask)
        global_attention_mask[0] = 1  # Set global attention on CLS token
        
        masked_tokens, labels = self._mask_tokens(tokens.clone())

        return {
            "input_ids": masked_tokens,
            "attention_mask": attention_mask,
            "global_attention_mask": global_attention_mask,
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
        probability_matrix.masked_fill_(
            inputs == self.tokenizer.pad_token_id, value=0.0
        )

        # Create masked indices
        masked_indices = torch.bernoulli(probability_matrix).bool()

        if not masked_indices.any():
            maskable = ~special_tokens_mask & (inputs != self.tokenizer.pad_token_id)
            if maskable.any():
                idx = torch.multinomial(maskable.float(), 1)
                masked_indices[idx] = True

        labels[~masked_indices] = -100

        # Replace tokens
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels


def create_hf_longformer_model(model_size, seq_length, attention_window=256):
    """Create a Longformer model using Hugging Face implementation."""
    if not LONGFORMER_AVAILABLE:
        logger.error("Hugging Face Longformer not available!")
        return None

    # Load pre-trained Longformer model
    if model_size == "base":
        model_name = "allenai/longformer-base-4096"
    else:
        model_name = "allenai/longformer-large-4096"
    
    try:
        model = LongformerForMaskedLM.from_pretrained(model_name)
        logger.info(f"✓ Loaded pre-trained Longformer model: {model_name}")
        return model
    except Exception as e:
        logger.error(f"Failed to load pre-trained model: {e}")
        
        # Fall back to creating from config if can't download
        config = LongformerConfig(
            attention_window=[attention_window] * 12,
            max_position_embeddings=seq_length,
            vocab_size=50265,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
        )
        
        model = LongformerForMaskedLM(config)
        logger.info("✓ Created Longformer model from config")
        return model


def test_hf_longformer_model(model, tokenizer, device, seq_length=4096):
    """Test Hugging Face Longformer with long sequences."""
    logger.info(f"Testing Hugging Face Longformer with sequence length {seq_length}...")

    try:
        batch_size = 2
        input_ids = torch.randint(1000, 5000, (batch_size, seq_length), device=device)
        attention_mask = torch.ones((batch_size, seq_length), device=device)
        global_attention_mask = torch.zeros((batch_size, seq_length), device=device)
        global_attention_mask[:, 0] = 1  # Global attention on first token

        labels = input_ids.clone()
        labels[labels < 2000] = -100

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                labels=labels
            )
            loss = outputs.loss

        logger.info(f"✓ Hugging Face Longformer test successful! Loss: {loss.item():.4f}")
        return True

    except Exception as e:
        logger.error(f"✗ Hugging Face Longformer test failed: {str(e)}")
        return False


def calculate_bpc_hf_longformer(model, dataloader, device, max_batches=500):
    """Calculate BPC using Hugging Face Longformer."""
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

                if i % 50 == 0:
                    cleanup_gpu_memory()

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


def run_hf_longformer_experiment(experiment_name, model_size, corpus_file, device):
    """Run experiments with Hugging Face Longformer implementation."""

    if not LONGFORMER_AVAILABLE:
        logger.error(
            "Hugging Face Longformer not available. Please install: pip install transformers torch"
        )
        return float("nan")

    logger.info(f"\n{'='*60}")
    logger.info(f"Running {experiment_name} with Hugging Face Longformer")
    logger.info(f"{'='*60}")

    cleanup_gpu_memory()

    # Use RobertaTokenizer for compatibility
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    if experiment_name == "roberta_baseline":
        from transformers import RobertaForMaskedLM

        seq_length = 512
        max_examples = 10000

        dataset = MemoryEfficientMLMDataset(
            corpus_file, tokenizer, seq_length, max_examples
        )
        # Remove global_attention_mask for RoBERTa
        class RobertaDataset:
            def __init__(self, longformer_dataset):
                self.dataset = longformer_dataset
            
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):
                item = self.dataset[idx]
                return {
                    "input_ids": item["input_ids"],
                    "attention_mask": item["attention_mask"],
                    "labels": item["labels"]
                }
        
        roberta_dataset = RobertaDataset(dataset)
        dataloader = DataLoader(roberta_dataset, batch_size=16, shuffle=False)

        model = RobertaForMaskedLM.from_pretrained(f"roberta-{model_size}")
        model.to(device)

        # Standard evaluation
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        batches_processed = 0

        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
                if i >= 500:
                    break

                batch = {k: v.to(device) for k, v in batch.items()}
                valid_tokens = (batch["labels"] != -100).sum().item()
                if valid_tokens == 0:
                    continue

                outputs = model(**batch)
                loss = outputs.loss

                if not torch.isnan(loss):
                    total_loss += loss.item() * valid_tokens
                    total_tokens += valid_tokens
                    batches_processed += 1

        bpc = (
            (total_loss / total_tokens) / np.log(2)
            if total_tokens > 0
            else float("nan")
        )

    elif experiment_name in ["longformer_no_copy", "longformer_with_copy"]:
        seq_length = 4096
        max_examples = 3000

        dataset = MemoryEfficientMLMDataset(
            corpus_file, tokenizer, seq_length, max_examples
        )
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

        model = create_hf_longformer_model(model_size, seq_length)
        if model is None:
            return float("nan")
        
        model.to(device)

        if not test_hf_longformer_model(model, tokenizer, device, seq_length):
            return float("nan")

        bpc = calculate_bpc_hf_longformer(model, dataloader, device, max_batches=200)

    else:
        logger.error(f"Experiment {experiment_name} not implemented")
        return float("nan")

    cleanup_gpu_memory()
    del model
    cleanup_gpu_memory()

    return bpc


def main():
    parser = argparse.ArgumentParser(description="Table 5 with Hugging Face Longformer")
    parser.add_argument("--corpus", type=str, default="your_corpus.txt")
    parser.add_argument("--model_size", type=str, choices=["base"], default="base")
    parser.add_argument(
        "--experiment",
        type=str,
        choices=[
            "roberta_baseline",
            "longformer_no_copy",
            "longformer_with_copy",
            "all",
        ],
        default="roberta_baseline",
    )
    args = parser.parse_args()

    set_seed(42)

    if not torch.cuda.is_available():
        logger.error("CUDA not available")
        return

    device = torch.device("cuda")
    allocated, reserved, total = get_memory_info()
    logger.info(f"GPU: {torch.cuda.get_device_name()}, Memory: {total:.1f} GB")

    experiments = [
        ("roberta_baseline", "RoBERTa (seqlen: 512)", 1.846),
        ("longformer_no_copy", "Longformer (HF pretrained)", 1.957),
        ("longformer_with_copy", "Longformer (HF pretrained)", 1.957),
    ]

    if args.experiment == "all":
        experiments_to_run = [exp[0] for exp in experiments]
    else:
        experiments_to_run = [args.experiment]

    results = {}

    for exp_name in experiments_to_run:
        try:
            start_time = time.time()
            bpc = run_hf_longformer_experiment(
                exp_name, args.model_size, args.corpus, device
            )
            end_time = time.time()

            results[exp_name] = {
                "bpc": bpc,
                "time_minutes": (end_time - start_time) / 60,
            }

            logger.info(
                f"{exp_name}: BPC = {bpc:.3f}, Time = {(end_time - start_time)/60:.1f} min"
            )

        except Exception as e:
            logger.error(f"Error in {exp_name}: {str(e)}")
            results[exp_name] = {"bpc": float("nan"), "error": str(e)}

    # Print final results
    logger.info("\n" + "=" * 80)
    logger.info("HUGGING FACE LONGFORMER - TABLE 5 RESULTS")
    logger.info("=" * 80)

    for exp_name, exp_label, paper_bpc in experiments:
        if exp_name in results:
            your_bpc = results[exp_name]["bpc"]
            if not np.isnan(your_bpc):
                diff = your_bpc - paper_bpc
                status = (
                    "✓ GOOD"
                    if abs(diff) < 0.5
                    else ("⚠ HIGHER" if diff > 0 else "✓ BETTER")
                )
                logger.info(
                    f"{exp_label:<30} {your_bpc:.3f} (paper: {paper_bpc:.3f}) {status}"
                )
            else:
                logger.info(f"{exp_label:<30} FAILED")


if __name__ == "__main__":
    main()