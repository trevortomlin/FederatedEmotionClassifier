import flwr as fl
from flwr.common import Metrics
import torch
from Config import *
from NeuralNetwork import NeuralNetwork
from LoadDataset import load_datasets
from FlowerClient import FlowerClient
from typing import List, Tuple
import torch
from TrainEncoder import train_local_encoder

def client_fn(cid: str, trainloaders, valloaders) -> FlowerClient:
    """Create a Flower clientokenizert representing a single organization."""

    net = NeuralNetwork().to(DEVICE)

    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    
    # Create a  single Flower client representing a single organization
    return FlowerClient(net, trainloader, valloader)

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def main():

    if not LOCAL_MODEL_PATH.exists():
        print("Local Encoder Model Does not exist. Training...")
        train_local_encoder()

    print(
        f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
    )

    trainloaders, valloaders, testloader = load_datasets()
    
    # Create FedAvg strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
        min_fit_clients=10,  # Never sample less than 10 clients for training
        min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
        min_available_clients=10,  # Wait until all 10 clients are available
        evaluate_metrics_aggregation_fn=weighted_average, # <-- pass the metric aggregation function
    )

    # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
    client_resources = None
    if DEVICE.type == "cuda":
        client_resources = {"num_gpus": 1}

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=lambda x: client_fn(x, trainloaders, valloaders),
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
        client_resources=client_resources,
    )

if __name__ == "__main__":
    main()