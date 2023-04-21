from collections import OrderedDict
from functools import partial
import itertools
from typing import List, Tuple
from datasets import load_dataset_builder
from datasets import get_dataset_split_names
from datasets import load_dataset
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer
from transformers import DistilBertTokenizer
from transformers import AutoModel
import numpy as np
from transformers import AutoModelForSequenceClassification
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# DATA_SET = "emotion"

# dataset = load_dataset(DATA_SET)

# model_checkpoint = "distilbert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# distilbert_tokenizer = DistilBertTokenizer.from_pretrained(model_checkpoint)

# def tokenize(batch):
#     return tokenizer(batch["text"], padding=True, truncation=True)

# emotions_encoded = dataset.map(tokenize, batched=True, batch_size=None)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = AutoModel.from_pretrained("distilbert-base-uncased-finetuned-emotion/checkpoint-500").to(device)

# def extract_hidden_states(batch):
#     inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
#     with torch.no_grad():
#         last_hidden_state = model(**inputs).last_hidden_state
#     return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}

# emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True)

# X_train = emotions_hidden["train"]["hidden_state"]
# y_train = emotions_hidden["train"]["label"]

# X_val = emotions_hidden["validation"]["hidden_state"]
# y_val = emotions_hidden["validation"]["label"]

# X_test = emotions_hidden["test"]["hidden_state"]
# y_test = emotions_hidden["test"]["label"]

# log_reg = LogisticRegression(max_iter=1000)
# log_reg.fit(X_train, y_train)
# score = log_reg.score(X_val, y_val)
# print(f"Logistic Regression: {score}")


# dummy_clf = DummyClassifier(strategy="most_frequent")
# dummy_clf.fit(X_train, y_train)
# score = dummy_clf.score(X_val, y_val)
# print(f"Dummy Classifier: {score}")

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class NeuralNetwork(nn.Module):

#     def __init__(self):
#         super(NeuralNetwork, self).__init__()
#         self.fc1 = nn.Linear(768, 6)
        

#     def forward(self, x):
#         x = self.fc1(x)
#         return x

# mymodel = NeuralNetwork()

# learning_rate = 0.1
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(mymodel.parameters(), lr=learning_rate)

# num_epochs = 2
# for epoch in range(num_epochs):
#     correct = 0
#     for i, (X, y) in enumerate(zip(X_train, y_train)):
#         y_pred = mymodel(X)

#         loss = loss_fn(y_pred, y)

#         correct += (torch.argmax(y_pred) == y).float().item()

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # if (i+1) % 4000 == 0:
#         #     print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(X_train)}], Loss: {loss.item():.4f}")
#         #     accuracy = correct / (i+1) * 100
#         #     print(f"Accuracy: {accuracy:.2f}")

# print("Training Complete")

# with torch.no_grad():
#     correct = 0
#     for i, (X, y) in enumerate(zip(X_val, y_val)):

#         correct += (torch.argmax(mymodel(X)) == y).float().item()

#     accuracy = correct / len(X_val) * 100
#     print(f"Accuracy: {accuracy:.2f}")

# def plot_confusion_matrix(y_preds, y_true, labels):
#     cm = confusion_matrix(y_true, y_preds, normalize="true")
#     fig, ax = plt.subplots(figsize=(6, 6))
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
#     disp.plot(cmap="Blues", ax=ax, colorbar=False, values_format=".2f")
#     plt.title("Normalized Confusion Matrix")
#     plt.show()

#y_preds = log_reg.predict(X_val)
#plot_confusion_matrix(y_preds, y_val, emotions_encoded["train"].features["label"].names)

# num_labels = 6
# model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels).to(device)

# from sklearn.metrics import accuracy_score, f1_score

# def compute_metrics(p):
#     preds, labels = p
#     preds = np.argmax(preds, axis=1)
#     return {
#         "accuracy": accuracy_score(labels, preds),
#         "f1": f1_score(labels, preds, average="weighted"),
#     }

# from transformers import Trainer, TrainingArguments

# batch_size = 64
# logging_steps = len(emotions_encoded["train"]) // batch_size
# model_name = f"{model_checkpoint}-finetuned-emotion"
# training_args = TrainingArguments(output_dir=model_name,
#                                   num_train_epochs=2,
#                                   learning_rate=2e-5,
#                                   per_device_train_batch_size=batch_size,
#                                   per_device_eval_batch_size=batch_size,
#                                   weight_decay=0.01,
#                                   evaluation_strategy="epoch",
#                                   disable_tqdm=False,
#                                   logging_steps=logging_steps,
#                                   log_level="error",
#                                 )

# trainer = Trainer(model=model,
#                     args=training_args,
#                     compute_metrics=compute_metrics,
#                     train_dataset=emotions_encoded["train"],
#                     eval_dataset=emotions_encoded["validation"],
#                     tokenizer=tokenizer)

# trainer.train()

import flwr as fl
from flwr.common import Metrics
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)

NUM_CLIENTS = 10
BATCH_SIZE = 32

DATA_SET = "emotion"

class CustomHiddenState(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def tokenize(tokenizer, batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

def extract_hidden_states(model, tokenizer, batch):
    inputs = {k: v.to(DEVICE) for k, v in batch.items() if k in tokenizer.model_input_names}
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
    return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}

def load_datasets():
    dataset = load_dataset(DATA_SET)

    model_checkpoint = "distilbert-base-uncased"

    distilbert_tokenizer = DistilBertTokenizer.from_pretrained(model_checkpoint)

    emotions_encoded = dataset.map(lambda x: tokenize(distilbert_tokenizer, x), batched=True, batch_size=None)

    model = AutoModel.from_pretrained("distilbert-base-uncased-finetuned-emotion/checkpoint-500").to(DEVICE)

    emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    emotions_hidden = emotions_encoded.map(lambda x: extract_hidden_states(model, distilbert_tokenizer, x), batched=True)

    X_train = emotions_hidden["train"]["hidden_state"]
    y_train = emotions_hidden["train"]["label"]

    X_val = emotions_hidden["validation"]["hidden_state"]
    y_val = emotions_hidden["validation"]["label"]

    X_test = emotions_hidden["test"]["hidden_state"]
    y_test = emotions_hidden["test"]["label"]

    train_data = CustomHiddenState(X_train, y_train)
    val_data = CustomHiddenState(X_val, y_val)
    test_data = CustomHiddenState(X_test, y_test)

    partition_size_train = len(train_data) // NUM_CLIENTS
    lengths_train = [partition_size_train] * NUM_CLIENTS

    partition_size_val = len(val_data) // NUM_CLIENTS
    lengths_val = [partition_size_val] * NUM_CLIENTS
    
    train_datasets = random_split(train_data, lengths_train, torch.Generator().manual_seed(42))
    val_datasets = random_split(val_data, lengths_val, torch.Generator().manual_seed(42))

    train_loaders = []
    val_loaders = []

    for ds in train_datasets:
        train_loaders.append(DataLoader(ds, batch_size=BATCH_SIZE))
    
    for ds in val_datasets:
        val_loaders.append(DataLoader(ds, batch_size=BATCH_SIZE))

    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    print(len(train_loaders))

    return train_loaders, val_loaders, test_loader,

trainloaders, valloaders, testloader = load_datasets()

class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(768, 6)
        
    def forward(self, x):
        x = self.fc1(x)
        return x

def train(net, trainloader, epochs: int, verbose=False):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for X, y in trainloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += y.size(0)
            correct += (torch.max(outputs.data, 1)[1] == y).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for X, y in testloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            outputs = net(X)
            loss += criterion(outputs, y).item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

# trainloader = trainloaders[0]
# valloader = valloaders[0]
# net = NeuralNetwork().to(DEVICE)

# for epoch in range(5):
#     train(net, trainloader, 1)
#     loss, accuracy = test(net, valloader)
#     print(f"Epoch {epoch+1}: validation loss {loss}, accuracy {accuracy}")

# loss, accuracy = test(net, testloader)
# print(f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}")

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
    
def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""

    # Load model
    net = NeuralNetwork().to(DEVICE)

    # Load data (CIFAR-10)
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
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
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
    client_resources=client_resources,
)