# download_openwebtext.py
import os
import argparse
from datasets import load_dataset
from tqdm import tqdm
import random

def main():
    parser = argparse.ArgumentParser(description='Download a sample of the OpenWebText dataset')
    parser.add_argument('--output_file', type=str, default='your_corpus.txt',
                      help='Output filename')
    parser.add_argument('--sample_size', type=int, default=5000,
                      help='Number of documents to sample')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Set the random seed
    random.seed(args.seed)
    
    print(f"Loading OpenWebText dataset from Hugging Face...")
    
    # Load a small sample of the OpenWebText dataset
    # By using streaming mode and taking a sample, we avoid downloading the entire dataset
    dataset = load_dataset("Skylion007/openwebtext", streaming=True, trust_remote_code=True)
    
    print(f"Sampling {args.sample_size} documents...")
    
    # We'll sample from the training split
    train_dataset = dataset["train"]
    
    # In streaming mode, we need to iterate through and sample manually
    # This approach avoids loading the entire dataset into memory
    sampled_docs = []
    for i, example in enumerate(tqdm(train_dataset, desc="Scanning dataset")):
        # Use reservoir sampling to get a random sample
        if len(sampled_docs) < args.sample_size:
            sampled_docs.append(example["text"])
        else:
            # Replace elements with decreasing probability
            j = random.randint(0, i)
            if j < args.sample_size:
                sampled_docs[j] = example["text"]
        
        # Stop after we've seen enough examples to ensure a good random sample
        if i >= args.sample_size * 20:
            break
    
    print(f"Writing {len(sampled_docs)} documents to {args.output_file}...")
    
    # Write the sampled documents to the output file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for doc in sampled_docs:
            # Format document with double newlines between documents for the MLM dataset
            f.write(doc + "\n\n")
    
    # Verify the file was created
    if os.path.exists(args.output_file):
        file_size_mb = os.path.getsize(args.output_file) / (1024 * 1024)
        print(f"Successfully created dataset ({file_size_mb:.2f} MB)")
    else:
        print(f"Error: Failed to create {args.output_file}")

if __name__ == "__main__":
    main()