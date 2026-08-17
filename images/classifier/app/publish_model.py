import argparse
import os
from huggingface_hub import HfApi

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, required=True, help="Hugging Face repo ID (e.g. org/model)")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained .skops model")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Warning: HF_TOKEN environment variable not set. Upload may fail if repo is private or requires auth.")

    api = HfApi(token=token)
    
    print(f"Ensuring repository {args.repo_id} exists...")
    try:
        api.create_repo(repo_id=args.repo_id, exist_ok=True, repo_type="model")
    except Exception as e:
        print(f"Could not create/verify repo: {e}")
        # Continue anyway, might just be lack of permissions to create, but can push

    print(f"Uploading {args.model_path} to {args.repo_id}...")
    try:
        api.upload_file(
            path_or_fileobj=args.model_path,
            path_in_repo=os.path.basename(args.model_path),
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Update BearID SVM model"
        )
        print("Upload complete.")
    except Exception as e:
        print(f"Failed to upload model: {e}")
        exit(1)

if __name__ == "__main__":
    main()
