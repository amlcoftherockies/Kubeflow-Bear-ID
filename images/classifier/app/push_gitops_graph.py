import argparse
import os
import tempfile
import git
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-url", type=str, required=True, help="GitOps repo URL")
    parser.add_argument("--branch", type=str, required=True, help="Git branch")
    parser.add_argument("--namespace", type=str, required=True, help="Target namespace")
    args = parser.parse_args()

    print(f"Cloning {args.repo_url} (branch {args.branch})...")
    
    # We use a temp directory to clone
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # We assume GIT_PAT is injected via K8s secret as env var in KFP
            repo_url = args.repo_url
            if "GIT_PAT" in os.environ:
                auth_url = repo_url.replace("https://", f"https://oauth2:{os.environ['GIT_PAT']}@")
                repo = git.Repo.clone_from(auth_url, tmpdir, branch=args.branch)
            else:
                repo = git.Repo.clone_from(repo_url, tmpdir, branch=args.branch)
                
            # Create KServe InferenceGraph YAML payload
            manifest = {
                "apiVersion": "serving.kserve.io/v1alpha1",
                "kind": "InferenceGraph",
                "metadata": {
                    "name": "bearid-graph",
                    "namespace": args.namespace
                },
                "spec": {
                    "nodes": {
                        "root": {
                            "routerType": "Sequence",
                            "steps": [
                                {"serviceName": "serve-preprocess"},
                                {"serviceName": "serve-embedder"},
                                {"serviceName": "serve-svm"}
                            ]
                        }
                    }
                }
            }
            
            manifest_path = os.path.join(tmpdir, "bearid-graph.yaml")
            with open(manifest_path, "w") as f:
                yaml.dump(manifest, f)
            
            print("Committing InferenceGraph to GitOps repo...")
            repo.index.add(["bearid-graph.yaml"])
            repo.index.commit("Update BearID InferenceGraph routing")
            repo.remotes.origin.push()
            print("GitOps push successful.")
            
        except Exception as e:
            print(f"Failed to push GitOps graph: {e}")

if __name__ == "__main__":
    main()
