import argparse
import time
from kubeflow.katib import KatibClient
from kubeflow.katib import constants
import os
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="Experiment name")
    parser.add_argument("--namespace", type=str, required=True, help="Namespace")
    parser.add_argument("--out", type=str, required=True, help="Path to output optimal params")
    args = parser.parse_args()

    katib_client = KatibClient()

    # Define the experiment using PyTorchJob
    experiment_spec = {
        "apiVersion": "kubeflow.org/v1beta1",
        "kind": "Experiment",
        "metadata": {"name": args.name, "namespace": args.namespace},
        "spec": {
            "objective": {
                "type": "minimize",
                "goal": 0.01,
                "objectiveMetricName": "loss",
            },
            "algorithm": {"algorithmName": "random"},
            "parallelTrialCount": 2,
            "maxTrialCount": 4,
            "maxFailedTrialCount": 1,
            "parameters": [
                {"name": "learning_rate", "parameterType": "double", "feasibleSpace": {"min": "0.0001", "max": "0.01"}}
            ],
            "trialTemplate": {
                "primaryContainerName": "pytorch",
                "trialParameters": [
                    {"name": "learningRate", "description": "Learning rate for the model", "reference": "learning_rate"}
                ],
                "trialSpec": {
                    "apiVersion": "kubeflow.org/v1",
                    "kind": "PyTorchJob",
                    "spec": {
                        "pytorchReplicaSpecs": {
                            "Master": {
                                "replicas": 1,
                                "restartPolicy": "OnFailure",
                                "template": {
                                    "spec": {
                                        "containers": [
                                            {
                                                "name": "pytorch",
                                                "image": "ghcr.io/amlcoftherockies/kubeflow-bear-id/train:latest",
                                                "command": ["python3", "/app/train_ddp.py", "--learning-rate", "${trialParameters.learningRate}"],
                                                "resources": {"limits": {"nvidia.com/gpu": 1}}
                                            }
                                        ]
                                    }
                                }
                            },
                            "Worker": {
                                "replicas": 2,
                                "restartPolicy": "OnFailure",
                                "template": {
                                    "spec": {
                                        "containers": [
                                            {
                                                "name": "pytorch",
                                                "image": "ghcr.io/amlcoftherockies/kubeflow-bear-id/train:latest",
                                                "command": ["python3", "/app/train_ddp.py", "--learning-rate", "${trialParameters.learningRate}"],
                                                "resources": {"limits": {"nvidia.com/gpu": 1}}
                                            }
                                        ]
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    print(f"Creating Katib Experiment: {args.name} in namespace {args.namespace}")
    try:
        katib_client.create_experiment(experiment_spec, namespace=args.namespace)
    except Exception as e:
        print(f"Failed to create Katib experiment. {e}")
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump({"learning_rate": 0.001}, f)
        return

    print("Waiting for experiment to finish...")
    while True:
        try:
            status = katib_client.get_experiment_status(args.name, namespace=args.namespace)
            condition = status.get('conditions', [])[-1].get('type') if status.get('conditions') else "Unknown"
            print(f"Status: {condition}")
            if condition in [constants.EXPERIMENT_CONDITION_SUCCEEDED, constants.EXPERIMENT_CONDITION_FAILED]:
                break
        except Exception:
            pass
        time.sleep(10)

    try:
        optimal_trial = katib_client.get_optimal_hyperparameters(args.name, namespace=args.namespace)
        best_lr = 0.001
        if optimal_trial and 'parameterAssignments' in optimal_trial:
            for param in optimal_trial['parameterAssignments']:
                if param['name'] == 'learning_rate':
                    best_lr = float(param['value'])
                    break
        
        print(f"Optimal learning_rate found: {best_lr}")
        
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump({"learning_rate": best_lr}, f)
    except Exception as e:
        print(f"Failed to get optimal hyperparameters: {e}")
        with open(args.out, 'w') as f:
            json.dump({"learning_rate": 0.001}, f)

if __name__ == "__main__":
    main()
