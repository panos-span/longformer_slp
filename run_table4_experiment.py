import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from longformer_experiment import (
    SimplifiedLongformer,
    SimplifiedLongformerConfig,
    Text8Dataset,
    evaluate_model,
)


def run_table4_experiment(args):
    """Reproduce Table 4 from the Longformer paper with memory-efficient settings."""

    print(f"Running experiment with configuration: {args.config_name}")
    print(f"Device: {args.device}")

    # Define the configurations for the experiment
    configs = {
        "decreasing": {
            "window_sizes": [256, 128, 64, 32, 16, 8],
            "dilations": [1, 1, 1, 1, 1, 1],
        },
        "fixed": {
            "window_sizes": [64, 64, 64, 64, 64, 64],
            "dilations": [1, 1, 1, 1, 1, 1],
        },
        "increasing": {
            "window_sizes": [8, 16, 32, 64, 128, 256],
            "dilations": [1, 1, 1, 1, 1, 1],
        },
        "no_dilation": {
            "window_sizes": [8, 16, 32, 64, 128, 256],
            "dilations": [1, 1, 1, 1, 1, 1],
        },
        "dilation_2_heads": {
            "window_sizes": [8, 16, 32, 64, 128, 256],
            "dilations": [1, 1, 1, 1, 2, 2],
        },
    }

    # Get the configuration for the experiment
    config_params = configs[args.config_name]

    # Create model configuration
    model_config = SimplifiedLongformerConfig(
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=1024,
        attention_window=config_params["window_sizes"],
        attention_dilation=config_params["dilations"],
        attention_mode="sliding_chunks",
        max_position_embeddings=512,
    )

    # Create the model
    model = SimplifiedLongformer(model_config)

    # Move model to the specified device
    device = torch.device(args.device)
    model.to(device)

    # Create dataset and dataloader for dev set
    dev_dataset = Text8Dataset(
        args.data_path, model_config.max_position_embeddings, split="dev"
    )
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")
    print(f"Window sizes: {config_params['window_sizes']}")
    print(f"Dilations: {config_params['dilations']}")

    # Train for a fixed number of steps
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Start timer
    start_time = time.time()

    # Training loop
    model.train()

    print(f"Training for {args.steps} steps...")
    step = 0

    # Train for specified number of steps
    for epoch in range(100):  # Large number to ensure we reach steps limit
        if step >= args.steps:
            break

        for batch in tqdm(dev_loader, desc=f"Epoch {epoch+1}"):
            if step >= args.steps:
                break

            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            _, loss = model(input_ids, labels=labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

            if step % 50 == 0:
                elapsed = time.time() - start_time
                print(f"Step {step} | Loss: {loss:.4f} | Time: {elapsed:.2f}s")

    # Evaluate the model
    print("Evaluating model...")
    eval_loss, eval_bpc = evaluate_model(model, dev_loader, device)

    # Print results
    print("=" * 40)
    print(f"Experiment: {args.config_name}")
    print(f"Dev BPC: {eval_bpc:.4f}")
    print("=" * 40)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    with open(
        os.path.join(args.output_dir, f"{args.config_name}_results.txt"), "w"
    ) as f:
        f.write(f"Config: {args.config_name}\n")
        f.write(f"Window sizes: {config_params['window_sizes']}\n")
        f.write(f"Dilations: {config_params['dilations']}\n")
        f.write(f"Dev BPC: {eval_bpc:.4f}\n")

    return eval_bpc


def run_all_configs(args):
    """Run experiments for all configurations."""
    configs = ["decreasing", "fixed", "increasing", "no_dilation", "dilation_2_heads"]
    results = {}

    for config_name in configs:
        args.config_name = config_name
        bpc = run_table4_experiment(args)
        results[config_name] = bpc

    # Print summary of all results
    print("\n" + "=" * 50)
    print("SUMMARY OF RESULTS")
    print("=" * 50)
    for config_name, bpc in results.items():
        print(f"{config_name.ljust(20)}: {bpc:.4f}")

    # Save summary
    with open(os.path.join(args.output_dir, "all_results.txt"), "w") as f:
        f.write("Summary of Table 4 experiment results:\n\n")
        for config_name, bpc in results.items():
            f.write(f"{config_name}: {bpc:.4f}\n")


def download_text8_dataset(output_path):
    """Download text8 dataset if it doesn't exist."""
    import urllib.request

    if os.path.exists(output_path):
        print(f"Dataset already exists at {output_path}")
        return

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Downloading text8 dataset...")
    url = "http://mattmahoney.net/dc/text8.zip"
    zip_path = output_path + ".zip"

    # Download the file
    urllib.request.urlretrieve(url, zip_path)

    # Extract the file
    print("Extracting dataset...")
    import zipfile

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(os.path.dirname(output_path))

    # Remove the zip file
    os.remove(zip_path)
    print(f"Dataset downloaded and extracted to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Longformer Table 4 experiments")
    parser.add_argument(
        "--config_name",
        type=str,
        default="all",
        choices=[
            "all",
            "decreasing",
            "fixed",
            "increasing",
            "no_dilation",
            "dilation_2_heads",
        ],
        help="Which configuration to use",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/text8",
        help="Path to the text8 dataset",
    )
    parser.add_argument(
        "--download_dataset",
        action="store_true",
        help="Download the text8 dataset if it doesn't exist",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results", help="Directory to save results"
    )
    parser.add_argument(
        "--steps", type=int, default=150000, help="Number of training steps"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-4, help="Learning rate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training (cuda or cpu)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Download dataset if requested
    if args.download_dataset:
        download_text8_dataset(args.data_path)

    # Run experiments
    if args.config_name == "all":
        run_all_configs(args)
    else:
        run_table4_experiment(args)


if __name__ == "__main__":
    main()
