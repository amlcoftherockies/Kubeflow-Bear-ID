#!/usr/bin/env python3
import argparse
import yaml
import sys

def main():
    parser = argparse.ArgumentParser(description="Template KServe InferenceService for GitOps")
    parser.add_argument("--repo-id", required=True, help="Hugging Face Model Repo ID (e.g., amlcoftherockies/bearid-svm)")
    parser.add_argument("--namespace", default="kubeflow-user", help="Target Kubernetes namespace")
    parser.add_argument("--min-replicas", default=1, type=int, help="Minimum pod replicas")
    parser.add_argument("--max-replicas", default=3, type=int, help="Maximum pod replicas")
    parser.add_argument("--out", default="inferenceservice.yaml", help="Output file path")

    args = parser.parse_args()

    # KServe InferenceService Python Dictionary
    manifest = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": "bearid-predictor",
            "namespace": args.namespace,
            "annotations": {
                "serving.kserve.io/minReplicas": str(args.min_replicas),
                "serving.kserve.io/maxReplicas": str(args.max_replicas)
            }
        },
        "spec": {
            "predictor": {
                "containers": [
                    {
                        "name": "kserve-container",
                        "image": "ghcr.io/amlcoftherockies/kubeflow-bear-id/serve:latest",
                        "ports": [{"containerPort": 8080, "protocol": "TCP"}],
                        "env": [
                            {"name": "MODEL_REPO_ID", "value": args.repo_id}
                        ],
                        "resources": {
                            "requests": {"cpu": "2", "memory": "4Gi"},
                            "limits": {"cpu": "4", "memory": "8Gi"}
                        }
                    }
                ]
            }
        }
    }

    # Dump the dictionary safely to standard Kubernetes YAML
    with open(args.out, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
    
    print(f"Successfully templated KServe manifest to {args.out}")

if __name__ == "__main__":
    main()