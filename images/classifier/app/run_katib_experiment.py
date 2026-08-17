import argparse
import time
from kubeflow.katib import KatibClient
from kubeflow.katib import constants

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-c-file", type=str, required=True)
    args = parser.parse_args()

    # Create Katib client
    # Assuming execution inside Kubeflow where RBAC allows Katib operations
    katib_client = KatibClient()

    # Define the experiment
    name = "bearid-svm-tuning-" + str(int(time.time()))
    namespace = "kubeflow-user-example-com" # Adjust if necessary or pass as arg
    
    experiment_spec = {
        "apiVersion": "kubeflow.org/v1beta1",
        "kind": "Experiment",
        "metadata": {"name": name, "namespace": namespace},
        "spec": {
            "objective": {
                "type": "maximize",
                "goal": 0.99,
                "objectiveMetricName": "accuracy",
            },
            "algorithm": {"algorithmName": "random"},
            "parallelTrialCount": 3,
            "maxTrialCount": 9,
            "maxFailedTrialCount": 3,
            "parameters": [
                {"name": "C", "parameterType": "double", "feasibleSpace": {"min": "0.1", "max": "10.0"}}
            ],
            "trialTemplate": {
                "primaryContainerName": "training-container",
                "trialParameters": [
                    {"name": "C", "description": "SVM C parameter", "reference": "C"}
                ],
                "trialSpec": {
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "spec": {
                        "template": {
                            "spec": {
                                "containers": [
                                    {
                                        "name": "training-container",
                                        "image": "ghcr.io/amlcoftherockies/kubeflow-bear-id/classifier:latest",
                                        "command": ["python", "train_svm_final.py", "--c-param", "${trialParameters.C}"]
                                    }
                                ],
                                "restartPolicy": "Never"
                            }
                        }
                    }
                }
            }
        }
    }

    print(f"Creating Katib Experiment: {name}")
    try:
        katib_client.create_experiment(experiment_spec, namespace=namespace)
    except Exception as e:
        print(f"Failed to create Katib experiment. {e}")
        # Write default fallback C if Katib fails
        with open(args.output_c_file, 'w') as f:
            f.write("1.0")
        return

    print("Waiting for experiment to finish...")
    # This is a naive polling loop, in a real KFP setup Katib handles waiting, 
    # but as requested we wait for it to complete.
    while True:
        try:
            status = katib_client.get_experiment_status(name, namespace=namespace)
            condition = status.get('conditions', [])[-1].get('type') if status.get('conditions') else "Unknown"
            print(f"Status: {condition}")
            if condition in [constants.EXPERIMENT_CONDITION_SUCCEEDED, constants.EXPERIMENT_CONDITION_FAILED]:
                break
        except Exception:
            pass
        time.sleep(10)

    try:
        optimal_trial = katib_client.get_optimal_hyperparameters(name, namespace=namespace)
        best_c = "1.0" # Default
        if optimal_trial and 'parameterAssignments' in optimal_trial:
            for param in optimal_trial['parameterAssignments']:
                if param['name'] == 'C':
                    best_c = param['value']
                    break
        
        print(f"Optimal C found: {best_c}")
        
        # Ensure output directory exists
        import os
        os.makedirs(os.path.dirname(args.output_c_file), exist_ok=True)
        with open(args.output_c_file, 'w') as f:
            f.write(best_c)
    except Exception as e:
        print(f"Failed to get optimal hyperparameters: {e}")
        # Write default fallback C
        with open(args.output_c_file, 'w') as f:
            f.write("1.0")

if __name__ == "__main__":
    main()
