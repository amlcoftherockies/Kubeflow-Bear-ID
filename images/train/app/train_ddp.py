import argparse
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.models import resnet34, ResNet34_Weights
import mlflow

# Dummy dataset for illustration (would read actual .pt files in production)
class BearDataset(Dataset):
    def __init__(self, size=100):
        self.size = size
        self.data = torch.randn(size, 3, 224, 224)
        self.labels = torch.randint(0, 10, (size,))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    # Initialize PyTorch DDP
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    print(f"Rank {dist.get_rank()} initializing ResNet-34 on {device}")
    
    # Load ResNet-34
    model = resnet34(weights=ResNet34_Weights.DEFAULT)
    
    # Freeze body
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace FC layer for 128D embeddings
    model.fc = nn.Linear(model.fc.in_features, 128)
    model = model.to(device)
    
    # Wrap in DDP
    if torch.cuda.is_available():
        model = DDP(model, device_ids=[local_rank])
    else:
        model = DDP(model)

    # Dataset & DataLoader
    dataset = BearDataset()
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.TripletMarginLoss()

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow.kubeflow.svc.cluster.local:5000"))
    
    # Only master node logs to MLflow
    is_master = dist.get_rank() == 0
    if is_master:
        mlflow.start_run()
        mlflow.log_params(vars(args))

    print(f"Rank {dist.get_rank()} starting training...")
    model.train()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.to(device)
            # In a real triplet scenario, we need Anchor, Positive, Negative.
            # Here we just mock the embeddings for the syntax.
            anchor = model(data)
            positive = model(data + torch.randn_like(data)*0.1)
            negative = model(torch.flip(data, dims=[3]))
            
            loss = criterion(anchor, positive, negative)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if is_master:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
            mlflow.log_metric("loss", loss.item(), step=epoch)

    if is_master:
        print("Logging final model to MLflow...")
        mlflow.pytorch.log_model(model.module, "resnet-bearid")
        mlflow.end_run()
        
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
