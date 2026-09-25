#!/usr/bin/env python3
"""
Helper script to publish all 4 delivered models to Hugging Face Hub.
Usage:
    python src/upload_models_to_hf.py --token YOUR_HF_TOKEN --username YOUR_USERNAME [--private]
"""

import os
import argparse
from huggingface_hub import HfApi, login
from transformers import AutoModel, AutoTokenizer

MODELS_CONFIG = [
    {
        "local_dir": "./delivered_model_agnews",
        "repo_name": "bert-base-agnews-delivered",
        "task": "Topic Classification (AG News)"
    },
    {
        "local_dir": "./delivered_model_ner",
        "repo_name": "bert-base-cased-conll2003-ner",
        "task": "Named Entity Recognition (CoNLL-2003)"
    },
    {
        "local_dir": "./delivered_model_pos",
        "repo_name": "bert-base-uncased-ud-ewt-pos",
        "task": "Part-of-Speech Tagging (UD English EWT)"
    },
    {
        "local_dir": "./delivered_model_qa",
        "repo_name": "bert-base-uncased-squad-qa",
        "task": "Extractive Question Answering (SQuAD v1.1)"
    }
]

def main():
    parser = argparse.ArgumentParser(description="Upload trained BERT models to Hugging Face Hub")
    parser.add_argument("--token", type=str, required=True, help="Your Hugging Face write token")
    parser.add_argument("--username", type=str, required=True, help="Your Hugging Face username")
    parser.add_argument("--private", action="store_true", help="Set repository visibility to private")
    args = parser.parse_args()

    login(token=args.token)
    api = HfApi()

    print(f"Logged in to Hugging Face as {args.username}")
    print(f"Target private mode: {args.private}")
    if args.private:
        print("Note: Remember to grant access to collaborator 'Dexterg83' on Hugging Face!")

    for item in MODELS_CONFIG:
        repo_id = f"{args.username}/{item['repo_name']}"
        local_dir = item["local_dir"]
        print(f"\n--- Uploading {item['task']} -> {repo_id} ---")
        if not os.path.exists(local_dir):
            print(f"Warning: Directory {local_dir} not found. Train and save the model first.")
            continue

        api.create_repo(repo_id=repo_id, private=args.private, exist_ok=True)
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            repo_type="model"
        )
        print(f"Successfully uploaded to https://huggingface.co/{repo_id}")

    print("\nAll available models have been processed.")

if __name__ == "__main__":
    main()
