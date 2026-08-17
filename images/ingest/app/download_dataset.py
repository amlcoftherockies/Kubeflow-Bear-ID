import argparse
import requests
import zipfile
import os
import io
import sys

def main():
    parser = argparse.ArgumentParser(description="Download and extract dataset")
    parser.add_argument("--dataset-url", type=str, required=True, help="URL to download the dataset ZIP")
    parser.add_argument("--output-path", type=str, required=True, help="Path to extract the dataset")
    args = parser.parse_args()

    print(f"Downloading dataset from {args.dataset_url}...")
    try:
        response = requests.get(args.dataset_url, stream=True)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to download dataset: {e}")
        sys.exit(1)

    print(f"Extracting to {args.output_path}...")
    os.makedirs(args.output_path, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(args.output_path)
    
    print("Download and extraction complete.")

if __name__ == "__main__":
    main()
