import argparse
import pandas as pd
from sklearn.svm import SVC
import skops.io as sio
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-file", type=str, required=False, help="Path to input embeddings CSV")
    parser.add_argument("--c-param", type=float, required=False, default=1.0, help="Explicit C parameter (e.g. from Katib trial)")
    parser.add_argument("--c-file", type=str, required=False, help="Path to file containing optimal C parameter")
    parser.add_argument("--model-output-path", type=str, required=True, help="Path to save the trained model .skops")
    args = parser.parse_args()

    c_value = args.c_param
    if args.c_file and os.path.exists(args.c_file):
        with open(args.c_file, 'r') as f:
            c_str = f.read().strip()
            if c_str:
                c_value = float(c_str)
    
    print(f"Using C parameter: {c_value}")

    # Dummy training block if no embeddings provided (e.g. during a raw Katib trial test)
    if not args.embeddings_file or not os.path.exists(args.embeddings_file):
        print("No embeddings file found. Generating dummy model.")
        X = [[0, 0], [1, 1]]
        y = [0, 1]
    else:
        print(f"Loading embeddings from {args.embeddings_file}...")
        df = pd.read_csv(args.embeddings_file, header=None)
        # BearID typically outputs ID in col 0, face img in col 1, then 128-D vector
        X = df.iloc[:, 2:].values
        y = df.iloc[:, 0].values

    print("Training Linear SVC...")
    model = SVC(kernel='linear', C=c_value, probability=True)
    model.fit(X, y)
    
    accuracy = model.score(X, y)
    print(f"accuracy={accuracy}")

    print(f"Exporting model to {args.model_output_path}...")
    os.makedirs(os.path.dirname(args.model_output_path), exist_ok=True)
    sio.dump(model, args.model_output_path)
    
    print("Training complete.")

if __name__ == "__main__":
    main()
